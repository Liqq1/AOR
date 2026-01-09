import re
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torchvision
from transformers.modeling_outputs import BaseModelOutputWithPast

from aor.constant import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN
from aor.models.layers import MLVLROIQueryModule
from llava.model.llava import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)


class SPILlavaLlamaModel(LlavaLlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_level_spi_features = 4
        dim = 1024
        self.dummy_size = 576

        self.spi_module = MLVLROIQueryModule(
            embed_dims=dim, out_dims=4096, num_levels=4
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        img_metas=None,
        bboxes=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        vision_tower = getattr(self, "vision_tower", None)
        if (
            vision_tower is not None
            and input_ids is not None
            and (input_ids.shape[1] != 1 or self.training)
            and images is not None
        ):
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(
                            image.unsqueeze(0), output_hidden_states=True
                        )
                        select_hidden_state_layer = getattr(
                            self.config, "mm_vision_select_layer", -1
                        )
                        select_hidden_state = image_forward_out.hidden_states[
                            select_hidden_state_layer
                        ]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_forward_outs = vision_tower(images, output_hidden_states=True)
                    select_hidden_state_layer = getattr(
                        self.config, "mm_vision_select_layer", -1
                    )
                    select_hidden_state = image_forward_outs.hidden_states[
                        select_hidden_state_layer
                    ]
                    image_features = select_hidden_state[:, 1:]

                    mlvl_spi_features = image_forward_outs.hidden_states[
                        select_hidden_state_layer::-3
                    ]
                    mlvl_spi_features = mlvl_spi_features[::-1]
                    mlvl_spi_features = mlvl_spi_features[
                        -self.num_level_spi_features:
                    ]
                    mlvl_spi_features = [item[:, 1:] for item in mlvl_spi_features]
            if bboxes is not None and (len(bboxes) > 0) and bboxes[0] is not None:
                valid_bboxes = True
            else:
                valid_bboxes = False
                bboxes = [
                    torch.as_tensor(
                        ((0, 0, 1, 1),),
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                    for _ in range(len(input_ids))
                ]
            assert not isinstance(images, list)
            self.cached_features = mlvl_spi_features

            mlvl_spi_features = self.spi_module(mlvl_spi_features, bboxes)
            if type(images) is list:
                image_features = [
                    self.mm_projector(image_feature)[0]
                    for image_feature in image_features
                ]
            else:
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(
                self.dummy_size,
                getattr(self.mm_projector, "in_features", None)
                or self.mm_projector[0].in_features,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0

            for cur_input_ids, cur_input_embeds, spi_feat in zip(
                input_ids, inputs_embeds, mlvl_spi_features
            ):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = (
                        cur_input_embeds + (0.0 * dummy_image_features).sum()
                    )
                    new_input_embeds.append(cur_input_embeds)
                    cur_image_idx += 1
                    continue

                cur_image_features = image_features[cur_image_idx]
                num_patches = cur_image_features.shape[0]
                if (
                    cur_input_ids == vision_tower.config.im_patch_token
                ).sum() != num_patches:
                    raise ValueError(
                        "The number of image patch tokens should be the same as the number of image patches."
                    )
                masked_indices = torch.where(
                    cur_input_ids == vision_tower.config.im_patch_token
                )[0]
                mask_index_start = masked_indices[0]
                if (
                    masked_indices
                    != torch.arange(
                        mask_index_start,
                        mask_index_start + num_patches,
                        device=masked_indices.device,
                        dtype=masked_indices.dtype,
                    )
                ).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                if orig_embeds_params is not None:
                    cur_new_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:mask_index_start].detach(),
                            cur_image_features,
                            cur_input_embeds[mask_index_start + num_patches:].detach(),
                        ),
                        dim=0,
                    )
                else:
                    cur_new_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:mask_index_start],
                            cur_image_features,
                            cur_input_embeds[mask_index_start + num_patches:],
                        ),
                        dim=0,
                    )
                cur_image_idx += 1
                if valid_bboxes:
                    spi_embeds = torch.zeros_like(cur_new_input_embeds)
                    spi_mask = (
                        cur_input_ids
                        == self.tokenizer.convert_tokens_to_ids(["<bbox>"])[0]
                    )
                    spi_embeds[spi_mask] = spi_feat.to(spi_embeds.dtype)

                    cur_new_input_embeds = (
                        cur_new_input_embeds
                        * (~spi_mask).to(cur_input_embeds.dtype)[:, None]
                        + spi_embeds
                    )
                else:
                    assert (
                        cur_input_ids
                        == self.tokenizer.convert_tokens_to_ids(["<bbox>"])[0]
                    ).sum() == 0, f"{(cur_input_ids == self.tokenizer.convert_tokens_to_ids(['<bbox>'])[0]).sum()},{self.tokenizer.decode(cur_input_ids)}"
                    cur_new_input_embeds = cur_new_input_embeds + spi_feat * 0.0
                new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LlavaLlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


def add_spatial_token(tokenizer):
    spi_tokens = ["<bbox>"]
    num_spi_tokens = tokenizer.add_tokens(spi_tokens, special_tokens=True)

    return tokenizer, num_spi_tokens


class SPILlavaMPTForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config):
        super(LlavaLlamaForCausalLM, self).__init__(config)
        self.model = SPILlavaLlamaModel(config)

        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.use_cot = True
        self.last_coord_idx = 0
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, *args, img_metas=None, bboxes=None, images=None, **kwargs):
        self.model.orig_forward = self.model.forward
        self.model.forward = partial(
            self.model.orig_forward, img_metas=img_metas, bboxes=bboxes
        )

        outputs = super().forward(*args, images=images, **kwargs)
        self.model.forward = self.model.orig_forward
        return outputs

    def initialize_vision_tokenizer(
        self,
        mm_use_im_start_end,
        tokenizer,
        device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    ):
        vision_config = self.get_model().vision_tower.config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        tokenizer, num_spi_tokens = add_spatial_token(tokenizer)
        if num_spi_tokens or mm_use_im_start_end:
            num_new_tokens = 0
            if mm_use_im_start_end:
                num_new_tokens = tokenizer.add_tokens(
                    [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
                )
                vision_config.im_start_token, vision_config.im_end_token = (
                    tokenizer.convert_tokens_to_ids(
                        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
                    )
                )
            self.resize_token_embeddings(len(tokenizer))

            num_new_tokens = num_new_tokens + num_spi_tokens
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)
                ]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                num_new_tokens = num_new_tokens - num_spi_tokens
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
        vision_config.bbox_token = tokenizer.convert_tokens_to_ids(["<bbox>"])[0]
        # broadcast the tokenizer to all modules
        for m in self.modules():
            m.tokenizer = tokenizer
        self.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
        self.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
        self.boc_token_id_list = (
            tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN),
            tokenizer.encode(DEFAULT_BOC_TOKEN, add_special_tokens=False)[0],
        )
        self.eoc_token_id_list = (
            tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN),
            tokenizer.encode(DEFAULT_EOC_TOKEN, add_special_tokens=False)[0],
        )
        self.bbox_token_id = vision_config.bbox_token

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if self.use_cot:
            original_input_ids = input_ids.clone()
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )

        if self.use_cot and "input_ids" in model_inputs:
            new_token_ids = model_inputs["input_ids"][:, -1:]
            find_eoc_token_id = False
            if new_token_ids in self.eoc_token_id_list:
                find_eoc_token_id = True
            if find_eoc_token_id:
                # need bounding box detection and use box align
                next_inputs_embeds, query_len = self.generate_box(
                    model_inputs, original_input_ids
                )
                if query_len != 0:
                    model_inputs["input_ids"] = None
                    model_inputs["inputs_embeds"] = next_inputs_embeds

        return model_inputs

    def generate_box(self, model_inputs, original_input_ids):
        assert original_input_ids.shape[0] == 1
        valid_start_ind = torch.where(
            torch.isin(
                original_input_ids,
                torch.as_tensor(
                    self.boc_token_id_list, device=original_input_ids.device
                ),
            )
        )[1].tolist()
        if not valid_start_ind:
            current_box_text = self.tokenizer.decode(
                original_input_ids[0, self.last_coord_idx:]
            )
            bbox = self.try_to_extract_box_str(current_box_text)
        else:
            valid_start_ind = valid_start_ind[-1]
            self.last_coord_idx = valid_start_ind
            current_box_text = self.tokenizer.decode(
                original_input_ids[0, valid_start_ind:]
            )
            bbox = self.extract_box_str(current_box_text)
        if bbox is None:
            return None, 0
        current_box = torch.tensor(
            bbox, dtype=self.dtype, device=self.model.cached_features[0].device
        )
        if current_box is None:
            print("fail to detect correct box from {}".format(current_box_text))
            raise ValueError
        box_feat = []
        mlvl_spi_feature_roi = torchvision.ops.roi_align(
            input=self.model.cached_features.unsqueeze(0).to(dtype=torch.float32),
            boxes=[current_box.to(dtype=torch.float32) * self.img_size],
            output_size=self.roi_align_output_size,
            sampling_ratio=2,
            spatial_scale=1 / self.roi_align_output_size,
        ).to(dtype=self.dtype)
        box_feat = (
            torch.nn.functional.adaptive_avg_pool2d(mlvl_spi_feature_roi, (1, 1))
            .squeeze(3)
            .squeeze(2)
        )
        init_inputs_embeds = self.model.embed_tokens(model_inputs["input_ids"])
        next_inputs_embeds = torch.cat(
            [init_inputs_embeds, box_feat.unsqueeze(0)], dim=1
        )
        return next_inputs_embeds, box_feat.shape[0]

    def try_to_extract_box_str(self, output):
        pattern = f"\s*[0-1]\.[0-9]+\s*,\s*[0-1]\.[0-9]+\s*,\s*[0-1]\.[0-9]+\s*,\s*[0-1]\.[0-9]+\s*\{DEFAULT_EOC_TOKEN}"
        boxes = re.findall(pattern, output)
        tmp_box = []
        for b in boxes:
            tmp_box.extend([float(k) for k in b[1:-1].split(",")])
        if len(tmp_box) >= 4:
            return tmp_box[-4:]
        return None

    def extract_box_str(self, output, mistral=False):
        pattern = f"\{DEFAULT_BOC_TOKEN}\s*[0-1]\.[0-9]+\s*,\s*[0-1]\.[0-9]+\s*,\s*[0-1]\.[0-9]+\s*,\s*[0-1]\.[0-9]+\s*\{DEFAULT_EOC_TOKEN}"

        boxes = re.findall(pattern, output)
        tmp_box = []
        for b in boxes:
            if mistral:
                tmp_box.extend([float(k) for k in b[1:-1].split(",")])
            else:
                tmp_box.extend([float(k) for k in b[1:-1].split(",")])
        if len(tmp_box) > 4:
            tmp_box = tmp_box[:4]
        elif len(tmp_box) != 4:
            return None
        assert len(tmp_box) == 4
        return tmp_box
