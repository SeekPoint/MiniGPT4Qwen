import os
import json

from PIL import Image

from collections import OrderedDict
from pathlib import Path

from lavis.datasets.datasets.minigpt4qwen_datasets import Minigpt4QwenDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image_id"]+'.jpg',
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class InstructionDataset(Minigpt4QwenDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            print(f"ann_path={ann_path}")
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]
        # print(f"self.vis_root={self.vis_root}")
        # self.vis_root=/ds/MiniGPT4Qwen/lavis/../cache/dataset/minigpt4/image
        image_path = os.path.join(self.vis_root,ann['image'])

        # print(f"image_path={image_path}")
        # image_path=/ds/MiniGPT4Qwen/lavis/../cache/dataset/minigpt4/image/2461.jpg
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        instruction = self.text_processor(ann['instruction'])

        output = ann['output']

        conversations = [
            {"from": "user", "value":instruction},
            {"from": "assistant", "value": output},
        ]

        return {
            "image": image,
            "conversations": conversations,
        }
