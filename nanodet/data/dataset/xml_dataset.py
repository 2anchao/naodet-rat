# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import glob
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

from pycocotools.coco import COCO

from nanodet.data.dataset.coco import CocoDataset


def get_file_list(path, type=".xml"):
    file_path = os.path.join(path, "*/*/*.xml")
    file_names = glob.glob(file_path)
    assert len(file_names)>0, "没读到数据!"
    return file_names


class CocoXML(COCO):
    def __init__(self, annotation):
        """
        Constructor of Microsoft COCO helper class for
        reading and visualizing annotations.
        :param annotation: annotation dict
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        dataset = annotation
        assert type(dataset) == dict, "annotation file format {} not supported".format(
            type(dataset)
        )
        self.dataset = dataset
        self.createIndex()


class XMLDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        super(XMLDataset, self).__init__(**kwargs)


    def xml_to_coco(self, ann_path):
        """
        convert xml annotations to coco_api
        :param ann_path:
        :return:
        """
        logging.info("loading annotations into memory...")
        tic = time.time()
        ann_file_names = get_file_list(ann_path, type=".xml")
        logging.info("Found {} annotation files.".format(len(ann_file_names)))
        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(self.class_names):
            categories.append(
                {"supercategory": supercat, "id": idx + 1, "name": supercat}
            )
        ann_id = 1
        for idx, xml_name in enumerate(ann_file_names):
            tree = ET.parse(os.path.join(ann_path, xml_name))
            root = tree.getroot()
            inter_name = root.find("filename").text
            ext = os.path.splitext(inter_name)[-1]
            file_name = xml_name.replace(".xml", ext)
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            info = {
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": idx + 1,
            }
            image_info.append(info)
            for _object in root.findall("object"):
                category = _object.find("name").text
                if category not in self.class_names:
                    logging.warning(
                        "WARNING! {} is not in class_names! "
                        "Pass this box annotation.".format(category)
                    )
                    continue
                for cat in categories:
                    if category == cat["name"]:
                        cat_id = cat["id"]
                xmin = int(eval(_object.find("bndbox").find("xmin").text))
                ymin = int(eval(_object.find("bndbox").find("ymin").text))
                xmax = int(eval(_object.find("bndbox").find("xmax").text))
                ymax = int(eval(_object.find("bndbox").find("ymax").text))
                w = xmax - xmin
                h = ymax - ymin
                if w < 0 or h < 0:
                    logging.warning(
                        "WARNING! Find error data in file {}! Box w and "
                        "h should > 0. Pass this box annotation.".format(xml_name)
                    )
                    continue
                coco_box = [max(xmin, 0), max(ymin, 0), min(w, width), min(h, height)]
                ann = {
                    "image_id": idx + 1,
                    "bbox": coco_box,
                    "category_id": cat_id,
                    "iscrowd": 0,
                    "id": ann_id,
                    "area": coco_box[2] * coco_box[3],
                }
                annotations.append(ann)
                ann_id += 1

        coco_dict = {
            "images": image_info,
            "categories": categories,
            "annotations": annotations,
        }
        logging.info(
            "Load {} xml files and {} boxes".format(len(image_info), len(annotations))
        )
        logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
        return coco_dict

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'file_name': '000000000139.jpg',
          'height': 426,
          'width': 640,
          'id': 139},
         ...
        ]
        """
        coco_dict = self.xml_to_coco(ann_path)
        self.coco_api = CocoXML(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

if __name__ == "__main__":
    import torch
    import cv2
    import numpy as np
    from nanodet.data.dataset import build_dataset
    from nanodet.util import cfg, load_config
    from nanodet.data.collate import naive_collate

    load_config(cfg, "config/nanodet_custom_xml_dataset.yml")

    train_dataset = build_dataset(cfg.data.train, "train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    for batch_meta in train_dataloader:
        imgs = batch_meta["img"]
        gt_bboxes = batch_meta["gt_bboxes"]
        gt_labels = batch_meta["gt_labels"]
        for labels, bboxes, img in zip(gt_labels, gt_bboxes, imgs):
            cv_img = np.asarray(img.cpu()).transpose(1, 2, 0)
            mean = np.array([103.53, 116.28, 123.675])
            std = np.array([57.375, 57.12, 58.395])
            cv_img *= std
            cv_img += mean
            cv_img = cv_img.astype(np.uint8)[:, :, [2, 1, 0]]
            cv_img = cv2.cvtColor(np.asarray(cv_img), cv2.COLOR_RGB2BGR)
            for i,box in enumerate(bboxes):
                label = labels[i]
                assert label==0
                xmin, ymin = box[0], box[1]
                xmax, ymax = box[2], box[3]
                cv2.rectangle(cv_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.imwrite("look.jpg", cv_img)
            import pdb; pdb.set_trace()


