import copy
import os

import torch
import json
import numpy as np
import re
from collections import defaultdict
import pickle

from aor.train.train import preprocess, preprocess_multimodal
from mmdet.datasets import CustomDataset
from aor.constant import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN, STR_BOX_TEMPLATE


pattern = re.compile(r"<(.*?)>")

GT_CLASS = {
    "right lung": 0,
    "right upper lung zone": 1,
    "right mid lung zone": 2,
    "right lower lung zone": 3,
    "right hilar structures": 4,
    "right apical zone": 5,
    "right costophrenic angle": 6,
    "right hemidiaphragm": 7,
    "left lung": 8,
    "left upper lung zone": 9,
    "left mid lung zone": 10,
    "left lower lung zone": 11,
    "left hilar structures": 12,
    "left apical zone": 13,
    "left costophrenic angle": 14,
    "left hemidiaphragm": 15,
    "trachea": 16,
    "spine": 17,
    "right clavicle": 18,
    "left clavicle": 19,
    "aortic arch": 20,
    "mediastinum": 21,
    "upper mediastinum": 22,
    "svc": 23,
    "cardiac silhouette": 24,
    "cavoatrial junction": 25,
    "right atrium": 26,
    "carina": 27,
    "abdomen": 28,
}

idx2class = {v: k for k, v in GT_CLASS.items()}


class Stage3Dataset(CustomDataset):
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

    BOX = (
        "cardiac silhouette",
        "cavoatrial junction",
        "right atrium",
        "mediastinum",
        "upper mediastinum",
        "aortic arch",
        "svc",
        "left lung",
        "left upper lung zone",
        "left mid lung zone",
        "left lower lung zone",
        "right lung",
        "right upper lung zone",
        "right mid lung zone",
        "right lower lung zone",
        "left clavicle",
        "right clavicle",
        "left hilar structures",
        "right hilar structures",
        "left costophrenic angle",
        "right costophrenic angle",
        "right apical zone",
        "right hemidiaphragm",
        "left apical zone",
        "left hemidiaphragm",
        "trachea",
        "spine",
        "carina",
        "abdomen",
    )

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
        img_size=224,
        ann_dir=None,
        data_type=None,
    ):
        self.multimodal_cfg = multimodal_cfg
        self.tokenizer = tokenizer
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.max_gt_per_img = max_gt_per_img
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.filter_small = filter_small
        self.is_test = test_mode
        self.patch_size = 14
        self.img_size = img_size
        assert img_size in [224, 336]
        self.data_type = data_type
        self.bbox_info = None
        self.box_mapping = {f"cot_bbox_{i+1}": n for i, n in enumerate(self.BOX)}
        self.split_box = {
            "cot_split_bbox_1": ["left", "cardiac silhouette"],
            "cot_split_bbox_2": ["right", "cardiac silhouette"],
            "cot_split_bbox_3": ["left", "mediastinum"],
            "cot_split_bbox_4": ["right", "mediastinum"],
            "cot_split_bbox_5": ["left", "upper mediastinum"],
            "cot_split_bbox_6": ["right", "upper mediastinum"],
        }
        keys = ["img", "ann_info"]
        if self.is_test:
            self.bbox_info = pickle.load(open(f"path/to/bbox_test", "rb"))
        else:
            self.bbox_info = pickle.load(open(f"path/to/bbox_train", "rb"))
        keys.extend(["gt_bboxes", "gt_labels"])
        img_norm_cfg = dict(
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            to_rgb=True,
        )
        train_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="RandomShift", shift_ratio=0.5, max_shift_px=32),
            dict(type="RandomFlip", flip_ratio=0.0),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=self.img_size),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=keys),
        ]

        test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_label=True),
            dict(type="RandomFlip", flip_ratio=0.0),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=self.img_size),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=keys),
        ]
        if test_mode:
            pipeline = test_pipeline
        else:
            pipeline = train_pipeline

        if test_mode:
            ann_file = os.path.join(ann_dir, "path/to/ann_cot_test")
            self.rg_ann_file = f"path/to/ann_region_test"
        else:
            ann_file = os.path.join(ann_dir, "path/to/ann_cot_train")
            self.rg_ann_file = f"path/to/ann_region_train"

        if self.is_test:
            label_path = os.path.join("path/to/reports_test")
        else:
            label_path = os.path.join("path/to/reports_train")
        with open(label_path, "rb") as f:
            self.labels = pickle.load(f)

        train = dict(
            ann_file=ann_file, img_prefix=vis_root, test_mode=False, pipeline=pipeline
        )

        super().__init__(**train)

        self.begin_str = "<image> provides an overview of the picture.\n"

        self.use_cot_str = os.getenv("USE_COT_STR", "")
        self.cot_text_str = "Anatomical structures are provided in the form of text.\n"
        self.cot_region_str = "Anatomical structures are provided in the form of region.\n"
        self.report_question = "Please generate a radiology report based on this Chest X-ray."
        self.region_report_question = "Please generate a corresponding radiology report based on this region of the Chest X-ray."

    def load_report_generation_annotations(self, json_data):
        if self.is_test:
            label_path = os.path.join("path/to/reports_test")
        else:
            label_path = os.path.join("path/to/reports_train")
        with open(label_path, "rb") as f:
            self.labels = pickle.load(f)

        key_prefix = "physionet.org/files/mimic-cxr-jpg/2.0.0/files/"

        image_data = defaultdict(dict)
        for i in json_data:
            if (
                i["image_path"] not in image_data
                and key_prefix + i["image_path"] in self.labels
            ):
                image_data[i["image_path"]] = i

        return [
            {"ann": i, "filename": i["image_path"], "type": "rg"}
            for i in image_data.values()
        ]

    def load_region_report_generation_annotations(self, json_data):
        if self.is_test:
            region_report_path = os.path.join("path/to/region_report_split_test")
        else:
            region_report_path = os.path.join("path/to/region_report_split_train")
        with open(region_report_path, "rb") as f:
            region_report = pickle.load(f)

        key_prefix = "physionet.org/files/mimic-cxr-jpg/2.0.0/files/"

        image_data = defaultdict(dict)
        for i in json_data:
            if (
                i["image_path"] not in image_data
                and (label_key := key_prefix + i["image_path"]) in region_report
            ):
                region_report_row = region_report[label_key]
                sentence = region_report_row["sentence"]
                original_bbox = region_report_row["original_bbox"]
                img_h = region_report_row["img_h"]
                img_w = region_report_row["img_w"]

                assert len(original_bbox) == len(sentence)

                scale_x = self.img_size / img_h  # img_w
                scale_y = self.img_size / img_w  # img_h

                i["sentence"] = [[sent] for sent in sentence]
                i["bbox"] = np.array(original_bbox) * np.array(
                    (scale_x, scale_y, scale_x, scale_y)
                )
                i["labels"] = np.arange(len(original_bbox))

                image_data[i["image_path"]] = i

        return [
            {"ann": i, "filename": i["image_path"], "type": "rrg"}
            for i in image_data.values()
        ]

    def load_annotations(self, ann_file):
        # cot_answer
        json_data = json.load(open(ann_file))
        cot_annos = [
            {
                "ann": [i["original_instance"]],
                "filename": i["original_instance"]["image_path"],
                "type": "cot",
            }
            for i in json_data
        ]

        # cot_answer_1
        json_data_1 = json.load(open(ann_file))
        cot_annos_1 = [
            {
                "ann": [i["original_instance"]],
                "filename": i["original_instance"]["image_path"],
                "type": "cot_1",
            }
            for i in json_data_1
        ]

        # cot_answer_2
        json_data_2 = json.load(open(ann_file))
        cot_annos_2 = [
            {
                "ann": [i["original_instance"]],
                "filename": i["original_instance"]["image_path"],
                "type": "cot_2",
            }
            for i in json_data_2
        ]

        json_data_rg = json.load(open(self.rg_ann_file))
        rg_annos = self.load_report_generation_annotations(json_data_rg)
        json_data_rrg = json.load(open(self.rg_ann_file))
        rrg_annos = self.load_region_report_generation_annotations(json_data_rrg)

        if self.data_type == "cot_answer":
            new_annos = [*cot_annos, *rg_annos, *rrg_annos]
        elif self.data_type == "cot_answer_1_2":
            new_annos = [*cot_annos_1, *cot_annos_2, *rg_annos, *rrg_annos]
        elif self.data_type == "cot_answer_1":
            new_annos = [*cot_annos_1, *rg_annos, *rrg_annos]
        elif self.data_type == "cot_answer_2":
            new_annos = [*cot_annos_2, *rg_annos, *rrg_annos]

        return new_annos

    def box2str(self, box):
        return (
            f"{DEFAULT_BOC_TOKEN}{STR_BOX_TEMPLATE.format(*box)}{DEFAULT_EOC_TOKEN}<b>"
        )

    def split_bbox(self, position, box):
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        l = w / 4
        if position == "right":
            xmin = max(xmin - l, 0)
            xmax = xmin + l
        else:
            xmin = xmax - l
            xmax = min(xmax + l, self.img_size)
        return (xmin, ymin, xmax, ymax)

    def process_answer(self, filename, answer: str):
        if not answer:
            return "None"
        box_info = self.bbox_info.get(filename)
        matches = pattern.findall(answer)
        boxes = []
        if box_info is None:
            print(0, filename)
            return answer, []

        for match in matches:
            if c := self.box_mapping.get(match, None):
                box = box_info.get(c, None)
                if box is None:
                    print(1, filename)
                    continue
                box = [i / self.img_size for i in box]
            elif match == "cot_merge_bbox_1":
                left_box = box_info.get("left lung", None)
                right_box = box_info.get("right lung", None)
                if left_box is None or right_box is None:
                    print(2, filename)
                    continue
                xmin = min(left_box[0], right_box[0]) / self.img_size
                ymin = min(left_box[1], right_box[1]) / self.img_size
                xmax = max(left_box[2], right_box[2]) / self.img_size
                ymax = max(left_box[3], right_box[3]) / self.img_size
                box = [xmin, ymin, xmax, ymax]
            elif info := self.split_box.get(match, None):
                b = box_info.get(info[1], None)
                if b is None:
                    print(3, filename)
                    continue
                box = self.split_bbox(info[0], b)
                box = [i / self.img_size for i in box]
            else:
                print(match, filename)
                raise RuntimeError()
            answer = answer.replace(f"<{match}>", self.box2str(box), 1)
            boxes.append(box)

        return answer, boxes

    def get_ann_info(self, idx):
        data = self.data_infos[idx]
        questions = []
        answers = []
        bboxes = []
        filename = data["filename"]
        data_indices = []
        answer_type = []
        data_type = data["type"]
        if data_type == "cot":
            for d in data["ann"]:
                question_bbox = d["question"]
                if item := self.bbox_info.get(filename, None):
                    if d["template_program"] == "program_47":
                        objs = ["cardiac silhouette"]
                    elif d["template_program"] == "program_48":
                        objs = ["upper mediastinum"]
                    else:
                        objs = list(d["template_arguments"]["object"].values())
                    bbox = []
                    for i in objs:
                        if box := item.get(i, None):
                            bbox.append([i / self.img_size for i in box])
                            question_bbox = re.sub(
                                rf"\b{i}\b", r"<b>", question_bbox, 1
                            )
                    if bbox:
                        bboxes.append(bbox)
                questions.append(question_bbox)
                answer, b = self.process_answer(filename, d["cot_answer"])
                answers.append(answer)
                if len(b) > 0 and not self.is_test:
                    bboxes.append(b)
                data_indices.append(d["idx"])
                answer_type.append(d["content_type"])

            ann = dict(
                questions="#".join(questions),
                answers="#".join(answers),
                idx=data_indices,
                answer_type=answer_type,
            )
            if bboxes:
                bb = []
                for b in bboxes:
                    bb.extend(b)
                ann["bboxes"] = np.array(bb)
        elif data_type == "cot_1":
            for d in data["ann"]:
                question_bbox = d["question"]
                questions.append(question_bbox)
                answer, b = self.process_answer(filename, d["cot_answer_1"])
                answers.append(answer)
                if len(b) > 0 and not self.is_test:
                    bboxes.append(b)
                data_indices.append(d["idx"])
                answer_type.append(d["content_type"])

            ann = dict(
                questions="#".join(questions),
                answers="#".join(answers),
                idx=data_indices,
                answer_type=answer_type,
            )
            if bboxes:
                bb = []
                for b in bboxes:
                    bb.extend(b)
                ann["bboxes"] = np.array(bb)
        elif data_type == "cot_2":
            for d in data["ann"]:
                question_bbox = d["question"]
                if item := self.bbox_info.get(filename, None):
                    if d["template_program"] == "program_47":
                        objs = ["cardiac silhouette"]
                    elif d["template_program"] == "program_48":
                        objs = ["upper mediastinum"]
                    else:
                        objs = list(d["template_arguments"]["object"].values())
                    bbox = []
                    for i in objs:
                        if box := item.get(i, None):
                            bbox.append([i / self.img_size for i in box])
                            question_bbox = re.sub(
                                rf"\b{i}\b", r"<b>", question_bbox, 1
                            )
                    if bbox:
                        bboxes.append(bbox)
                questions.append(question_bbox)
                answer, b = self.process_answer(filename, d["cot_answer_2"])
                answers.append(answer)
                if len(b) > 0 and not self.is_test:
                    bboxes.append(b)
                data_indices.append(d["idx"])
                answer_type.append(d["content_type"])

            ann = dict(
                questions="#".join(questions),
                answers="#".join(answers),
                idx=data_indices,
                answer_type=answer_type,
            )
            if bboxes:
                bb = []
                for b in bboxes:
                    bb.extend(b)
                ann["bboxes"] = np.array(bb)
        elif data_type == "rg":
            ann = dict(
                questions=data["ann"]["question"],
                answers=data["ann"]["answer"],
                idx=data["ann"]["idx"],
                answer_type=data["ann"]["content_type"],
            )
        elif data_type == "rrg":
            ann = dict(
                questions=data["ann"]["question"],
                answers=data["ann"]["answer"],
                idx=data["ann"]["idx"],
                answer_type=data["ann"]["content_type"],
                bboxes=data["ann"]["bbox"],
                labels=data["ann"]["labels"],
                sentence=data["ann"]["sentence"],
            )
            if self.is_test:
                ann["region_label"] = data["ann"]["region_label"]
        else:
            raise ValueError()
        ann["type"] = data_type
        ann.setdefault("labels", [])
        ann.setdefault("bboxes", np.zeros([0, 4]))
        return ann

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        data_type = data_item["ann_info"]["type"]
        if data_type == "rg":
            key = "physionet.org/files/mimic-cxr-jpg/2.0.0/files/" + data_item["img_metas"].data["ori_filename"]
            sent = self.get_caption(key)
            data_item["ann_info"]["questions"] = self.report_question
            data_item["ann_info"]["answers"] = sent
            return self.rg_train_process_test(data_item, data_type=data_type)
        elif data_type == "rrg":
            data_item["ann_info"]["questions"] = data_item["gt_bboxes"]
            data_item["ann_info"]["answers"] = data_item["ann_info"]["sentence"]
            data_item["ann_info"]["gt_labels"] = data_item["gt_labels"]
            return self.rrg_train_process_test(data_item, data_type=data_type)
        if isinstance(data_item["img"], list):
            data_item = {k: v[0] for k, v in data_item.items()}

        return self.train_process_test(data_item=data_item, data_type=data_type)

    def process_text(self, data_item):
        if isinstance(data_item["img"], list):
            data_item = {k: v[0] for k, v in data_item.items()}

        return self.train_process_test(data_item)

    def get_caption(self, key):
        sent = self.labels[key]
        return sent

    def rrg_train_process_test(self, data_item, data_type=None):
        image = data_item["img"].data
        ori_ans = data_item["ann_info"]["answers"]
        ori_que = data_item["ann_info"]["questions"].data.to(dtype=torch.float32)
        gt_labels = data_item["ann_info"]["gt_labels"].data
        if self.is_test:
            region_label = data_item["ann_info"]["region_label"]
        ori_ans = [ori_ans[i] for i in gt_labels]
        ori_que = ori_que / self.img_size
        assert (gt_labels == gt_labels.sort()[0]).all()
        if self.is_test:
            region_label = [region_label[i] for i in gt_labels]

        shuffle_ids = torch.randperm(len(ori_ans))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[: self.max_gt_per_img]
        ori_ans = [ori_ans[i] for i in shuffle_ids]
        if self.is_test:
            region_label = [region_label[i] for i in shuffle_ids]
        try:
            ori_que = ori_que[shuffle_ids]
        except IndexError:
            print("len(ori_ans)", len(ori_ans))
            print("ori_ans: ", ori_ans)
            print("len(ori_que)", len(ori_que))
            print("ori_que: ", ori_que)

        assert len(ori_ans) == len(ori_que)

        sources = dict()

        sources["conversations"] = []
        class_names = []
        cur_token_len = (image.shape[1] // self.patch_size) * (
            image.shape[2] // self.patch_size
        )

        bboxes = []
        for i in range(len(ori_ans)):
            question = "<bbox>"
            answer = " ".join(ori_ans[i])
            bboxes.append(ori_que[i])
            if i == 0:
                question = self.begin_str + self.region_report_question + question
            class_names.append(answer)
            sources["conversations"].append({"from": "human", "value": question})
            sources["conversations"].append({"from": "gpt", "value": answer})

        if self.is_test:
            sources["conversations"] = [sources["conversations"][0]]
        assert image.shape[1] == image.shape[2]
        sources = preprocess_multimodal(
            copy.deepcopy([sources["conversations"]]),
            self.multimodal_cfg,
            cur_token_len,
        )

        data_dict = preprocess(sources, self.tokenizer, is_test=self.is_test)
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        assert (
            data_dict["input_ids"]
            == self.tokenizer.convert_tokens_to_ids(["<bbox>"])[0]
        ).sum().item() == len(bboxes), data_item["ann_info"]

        data_dict["image"] = image
        data_dict["img_metas"] = data_item["img_metas"].data
        data_dict["answer"] = class_names
        if len(bboxes) > 0:
            data_dict["bboxes"] = torch.stack(bboxes)
        else:
            data_dict["bboxes"] = None
        data_dict["answer_type"] = data_item["ann_info"]["answer_type"]
        data_dict["idx"] = data_item["ann_info"]["idx"]
        data_dict["data_type"] = data_type
        if self.is_test:
            data_dict["region_label"] = region_label

        return data_dict

    def rg_train_process_test(self, data_item, data_type=None):
        image = data_item["img"].data
        ori_ans = [data_item["ann_info"]["answers"]]
        ori_que = [data_item["ann_info"]["questions"]]

        assert len(ori_ans) == len(ori_que)

        sources = dict()

        sources["conversations"] = []
        class_names = []
        cur_token_len = (image.shape[1] // self.patch_size) * (image.shape[2] // self.patch_size)

        for i in range(len(ori_ans)):
            question = ori_que[i]
            answer = ori_ans[i]
            if i == 0:
                question = self.begin_str + question
            class_names.append(answer)
            sources["conversations"].append({"from": "human", "value": question})
            sources["conversations"].append({"from": "gpt", "value": answer})

        if self.is_test:
            sources["conversations"] = [sources["conversations"][0]]
        assert image.shape[1] == image.shape[2]
        sources = preprocess_multimodal(
            copy.deepcopy([sources["conversations"]]),
            self.multimodal_cfg,
            cur_token_len,
        )

        data_dict = preprocess(sources, self.tokenizer, is_test=self.is_test)
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        data_dict["image"] = image
        data_dict["img_metas"] = data_item["img_metas"].data
        data_dict["answer"] = class_names
        data_dict["bboxes"] = None
        data_dict["answer_type"] = data_item["ann_info"]["answer_type"]
        data_dict["idx"] = data_item["ann_info"]["idx"]
        data_dict["data_type"] = data_type
        return data_dict

    def train_process_test(self, data_item, data_type=None):
        image = data_item["img"].data
        ori_ans = data_item["ann_info"]["answers"].split("#")
        ori_que = data_item["ann_info"]["questions"].split("#")
        bboxes = data_item["gt_bboxes"].data.to(dtype=torch.float32)
        assert len(ori_ans) == len(ori_que)
        sources = dict()

        sources["conversations"] = []
        class_names = []
        cur_token_len = (image.shape[1] // self.patch_size) * (image.shape[2] // self.patch_size)
        bbox_idx = 0
        for i in range(len(ori_ans)):
            question = ori_que[i]
            while "<b>" in question:
                question = question.replace("<b>", "<bbox>", 1)
                bbox_idx += 1

            if i == 0:
                if self.use_cot_str == "":
                    question = self.begin_str + question
                elif self.use_cot_str == "text_region":
                    if data_type == "cot_1":
                        question = self.begin_str + self.cot_text_str + question
                    elif data_type == "cot_2":
                        question = self.begin_str + self.cot_region_str + question
                    else:
                        question = self.begin_str + question
            sources["conversations"].append({"from": "human", "value": question})
            answer = ori_ans[i]
            if not self.is_test:
                while "<b>" in answer:
                    answer = answer.replace("<b>", "<bbox>", 1)
                    bbox_idx += 1
                sources["conversations"].append({"from": "gpt", "value": answer})
            class_names.append(answer)
        if bboxes is not None:
            if bboxes.shape[0] != bbox_idx:
                print(f'{bboxes.shape[0]}, {bbox_idx}, {data_item["ann_info"]["idx"]}, {question}, {answer}')
            assert (
                bboxes.shape[0] == bbox_idx
            ), f'{bboxes.shape[0]}, {bbox_idx}, {data_item["ann_info"]["idx"]}, {question}, {answer}'

        if self.is_test:
            sources["conversations"] = [sources["conversations"][0]]
        assert image.shape[1] == image.shape[2]
        sources = preprocess_multimodal(
            copy.deepcopy([sources["conversations"]]),
            self.multimodal_cfg,
            cur_token_len,
        )

        data_dict = preprocess(sources, self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        assert (
            data_dict["input_ids"]
            == self.tokenizer.convert_tokens_to_ids(["<bbox>"])[0]
        ).sum().item() == len(bboxes), data_item["ann_info"]

        data_dict["image"] = image
        data_dict["img_metas"] = data_item["img_metas"].data
        data_dict["answer"] = class_names
        data_dict["bboxes"] = bboxes if bboxes.shape[0] > 0 else None
        data_dict["answer_type"] = data_item["ann_info"]["answer_type"]
        data_dict["idx"] = data_item["ann_info"]["idx"]
        data_dict["data_type"] = data_type
        return data_dict

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)
