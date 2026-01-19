import copy
import os

import torch
import json
import numpy as np

from aor.train.train import preprocess, preprocess_multimodal
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from aor.constant import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN, STR_BOX_TEMPLATE


class Stage12Dataset(CocoDataset):
    CLASSES = (
        "right lung",
        "right upper lung zone",
        "right mid lung zone",
        "right lower lung zone",
        "right hilar structures",
        "right apical zone",
        "right hemidiaphragm",
        "left lung",
        "left upper lung zone",
        "left mid lung zone",
        "left lower lung zone",
        "left hilar structures",
        "left apical zone",
        "left costophrenic angle",
        "left hemidiaphragm",
        "trachea",
        "spine",
        "right clavicle",
        "left clavicle",
        "aortic arch",
        "mediastinum",
        "upper mediastinum",
        "svc",
        "cardiac silhouette",
        "cavoatrial junction",
        "right atrium",
        "abdomen",
        "right costophrenic angle",
        "carina",
    )

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        if self.is_test:
            ehrxqa_file = "path/to/ehrxqa_file_test"
        else:
            ehrxqa_file = "path/to/ehrxqa_file_train"
        ehrxqa_data = json.load(open(ehrxqa_file, "r"))
        ehrxqa_files = set()
        for item in ehrxqa_data:
            ehrxqa_files.add(item["image_path"])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        new_img_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info["filename"] = info["file_name"]
            if info["filename"] not in ehrxqa_files:
                continue
            new_img_ids.append(i)
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f"Annotation ids in '{ann_file}' are not unique!"
        new_data_infos = []
        img_ids = []
        for qs in self.question_type:
            for i, d in enumerate(data_infos):
                d_copy = copy.deepcopy(d)
                d_copy["question_type"] = qs
                new_data_infos.append(d_copy)
                img_ids.append(new_img_ids[i])
        if os.environ.get("PROB", "0") == "1":
            size = int(len(new_img_ids) * 0.8)
            img_ids = img_ids[:-size]
            new_data_infos = new_data_infos[:-size]
        assert len(img_ids) == len(new_data_infos)
        zipped = list(zip(img_ids, new_data_infos))
        img_ids, new_data_infos = zip(*zipped)

        self.img_ids = list(img_ids)
        return list(new_data_infos)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        box_ids = []
        for i, ann in enumerate(ann_info):
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            x1 = x1 * self.img_size / img_info["width"]
            w = w * self.img_size / img_info["width"]
            y1 = y1 * self.img_size / img_info["height"]
            h = h * self.img_size / img_info["height"]
            ann["bbox"] = x1, y1, w, h
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks_ann.append(ann.get("segmentation", None))
            box_ids.append(ann["id"])
        img_info["height"] = self.img_size
        img_info["width"] = self.img_size
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info["filename"].replace("jpg", "png")

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            question_type=img_info["question_type"],
            box_ids=box_ids,
        )

        return ann

    def __init__(
        self,
        tokenizer,
        multimodal_cfg=None,
        vis_processor=None,
        vis_root=None,
        add_eos=True,
        ignore_instruction=True,
        filter_small=False,
        test_mode=False,
        max_gt_per_img=100,
        ann_file=None,
    ):
        self.multimodal_cfg = multimodal_cfg
        self.tokenizer = tokenizer
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.max_gt_per_img = max_gt_per_img if not test_mode else 100
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.filter_small = filter_small
        self.is_test = test_mode
        self.patch_size = 14
        self.img_size = 336

        img_norm_cfg = dict(
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            to_rgb=True,
        )

        train_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="Resize", img_scale=(self.img_size, self.img_size), keep_ratio=False),
            dict(type="RandomShift", shift_ratio=0.5, max_shift_px=32),
            dict(type="FilterAnnotations", min_gt_bbox_wh=(2.0, 2.0)),
            dict(type="RandomFlip", flip_ratio=0.0),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=self.img_size),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "ann_info"]),
        ]

        test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="Resize", img_scale=(self.img_size, self.img_size), keep_ratio=False),
            dict(type="FilterAnnotations", min_gt_bbox_wh=(2.0, 2.0)),
            dict(type="RandomFlip", flip_ratio=0.0),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=self.img_size),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "ann_info"]),
        ]
        if test_mode:
            pipeline = test_pipeline
        else:
            pipeline = train_pipeline

        if ann_file is None:
            if self.is_test:
                ann_file = "path/to/ann_test"
            else:
                ann_file = "path/to/ann_train"

        img_prefix = self.vis_root
        train = dict(
            ann_file=ann_file, img_prefix=img_prefix, test_mode=False, pipeline=pipeline
        )
        question_type = os.environ.get("QUESTION")
        self.question_type = [int(i) for i in question_type]
        super(CocoDataset, self).__init__(**train)
        self.num_classes = len(self.CLASSES)
        self.begin_strs = [
            (
                "<image>\n In the conversation below, you are required to answer the category name based on what you see"
                " in the imagery inside a particular region. I will give you only one region each time. "
                f"Categories containing {', '.join(self.CLASSES)}.\n"
            ),
            (
                "<image>\n In the conversation below, you are required to answer the category name based on the given region. "
                f"The region is provided in coordinate form, {DEFAULT_BOC_TOKEN}x1, y1, x2, y2{DEFAULT_EOC_TOKEN}, with floating point number from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. "
                f"Categories containing {', '.join(self.CLASSES)}.\n"
            ),
            (
                "<image>\n In the conversation below, you are required to locate the corresponding region of the given category name in the image, "
                f"and output its coordinates in the form of {DEFAULT_BOC_TOKEN}x1, y1, x2, y2{DEFAULT_EOC_TOKEN}, with floating point number from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. I will give you only one category name each time.\n"
            ),
            (
                "<image>\n In the conversation below, you are required to answer the category name based on the given region. "
                f"The region is provided in coordinate form and imagery form. The coordinate form is {DEFAULT_BOC_TOKEN}x1, y1, x2, y2{DEFAULT_EOC_TOKEN}, with floating point number from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. "
                f"Categories containing {', '.join(self.CLASSES)}.\n"
            ),
        ]

    def train_process_test(self, data_item):
        image = data_item["img"].data
        ori_labels = data_item["gt_labels"].data
        ori_bboxes = data_item["gt_bboxes"].data

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[: self.max_gt_per_img]
        ori_bboxes = ori_bboxes[shuffle_ids]
        ori_labels = ori_labels[shuffle_ids]

        sources = dict()

        sources["conversations"] = []
        class_names = []
        class_labels = []
        cur_token_len = (image.shape[1] // self.patch_size) * (image.shape[2] // self.patch_size)

        ori_bboxes = copy.deepcopy(ori_bboxes) / image.shape[1]
        bboxes = []

        question_type = data_item["ann_info"]["question_type"]
        for i in range(len(ori_labels)):
            if question_type == 0:
                question = "<bbox>"
                answer = self.CLASSES[ori_labels[i]]
                bboxes.append(ori_bboxes[i])
            elif question_type == 1:
                question = f"{DEFAULT_BOC_TOKEN}{STR_BOX_TEMPLATE.format(*ori_bboxes[i].tolist())}{DEFAULT_EOC_TOKEN}"
                answer = self.CLASSES[ori_labels[i]]
            elif question_type == 2:
                question = self.CLASSES[ori_labels[i]]
                answer = f"{DEFAULT_BOC_TOKEN}{STR_BOX_TEMPLATE.format(*ori_bboxes[i].tolist())}{DEFAULT_EOC_TOKEN}"
            elif question_type == 3:
                question = f"{DEFAULT_BOC_TOKEN}{STR_BOX_TEMPLATE.format(*ori_bboxes[i].tolist())}{DEFAULT_EOC_TOKEN}<bbox>"
                answer = self.CLASSES[ori_labels[i]]
                bboxes.append(ori_bboxes[i])
            else:
                raise RuntimeError()
            if i == 0 or (self.is_test and question_type in [1, 2, 3]):
                question = self.begin_strs[question_type] + question
            class_names.append(answer)
            class_labels.append(self.CLASSES[ori_labels[i]])
            sources["conversations"].append({"from": "human", "value": question})
            if not self.is_test:
                sources["conversations"].append({"from": "gpt", "value": answer})

        assert image.shape[1] == image.shape[2]
        if not self.is_test or question_type == 0:
            if self.is_test:
                sources["conversations"] = [sources["conversations"][0]]
            sources = preprocess_multimodal(
                copy.deepcopy([sources["conversations"]]),
                self.multimodal_cfg,
                cur_token_len,
            )
            data_dict = preprocess(
                sources, self.tokenizer, question_type, is_test=self.is_test
            )
            if isinstance(i, int):
                data_dict = dict(
                    input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
                )
        else:
            input_ids = []
            labels = []
            for i in range(len(sources["conversations"])):
                s = {"conversations": [sources["conversations"][i]]}
                s = preprocess_multimodal(
                    copy.deepcopy([s["conversations"]]),
                    self.multimodal_cfg,
                    cur_token_len,
                )

                d = preprocess(s, self.tokenizer, is_test=self.is_test)
                input_ids.append(d["input_ids"][0])
                labels.append(d["labels"][0])
            data_dict = dict(input_ids=input_ids, labels=labels)
        if len(bboxes) > 0:
            data_dict["bboxes"] = torch.stack(bboxes)
        else:
            data_dict["bboxes"] = None
        data_dict["image"] = image
        data_dict["img_metas"] = data_item["img_metas"].data
        data_dict["class_names"] = class_names
        data_dict["class_labels"] = class_labels

        return data_dict

    def process_text(self, data_item):
        if isinstance(data_item["img"], list):
            data_item = {k: v[0] for k, v in data_item.items()}

        return self.train_process_test(data_item)

    def tokenize(self, text):
        t = text["instruction"]
        if not self.test_mode:
            t += text["answer"]
        res = self.tokenizer(
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )

        if (
            res["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(res["input_ids"]) < 512
            and self.add_eos
        ):
            res["input_ids"].append(self.tokenizer.eos_token_id)
            res["attention_mask"].append(1)
        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            bbox_index = labels.index(self.tokenizer.encode("<bbox>")[1])
            labels[:bbox_index] = [-100] * bbox_index

        res.update(labels=labels)
        return res

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)

        data_dict = self.process_text(data_item=data_item)

        return data_dict

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)
