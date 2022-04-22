import json
import os
import time

import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

load_config(cfg, "/project/train/src_repo/naodet-rat/config/nanodet_custom_lulu_xml_dataset.yml")
model_path = "/project/train/models/model_best/nanodet_model_best.pth" 
pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
local_rank = 0
logger = Logger(local_rank, use_tensorboard=False)

def init():
    """
    Initialize model
    Returns: model
    """
    device="cuda:0"
    model = build_model(cfg.model)
    ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, ckpt, logger)
    model = model.to(device).eval()
    return model

def process_image(handle = None , input_image = None , args = None , **kwargs):
    """
    Do inference to analysis input_image and get output
    Attributes:
        handle: algorithm handle returned by init()
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
    Returns: process result
    """
    # Process image here
    device="cuda:0"
    score_thresh = 0.1
    handle.to(device).eval()
    img_info = {"id": 0}
    img_info["file_name"] = None
    height, width = input_image.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    meta = dict(img_info=img_info, raw_img=input_image, img=input_image)
    meta = pipeline(None, meta, cfg.data.val.input_size)
    meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(device)
    meta = naive_collate([meta])
    meta["img"] = stack_batch_img(meta["img"], divisible=32)
    with torch.no_grad():
        results = handle.inference(meta)
        dets = results[0]
    # ori_img = meta["raw_img"][0]
    # import pdb; pdb.set_trace()
    result = {}
    result ["algorithm_data"] = {
        "is_alert": False,
        "target_count": 0,
        "target_info": []
        }

    result ["model_data"] = {
        "objects": []
        }
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                info = {
                "x":   x0,
                "y":   y0,
                "width":  x1-x0,
                "height":   y1-y0,
                "confidence":  score,
                "name": "rat"
                }
                result ["algorithm_data"]["target_info"].append(info)
                result ["algorithm_data"]["is_alert"] = True
                result ["algorithm_data"]["target_count"] += 1
                o_info = {
                "x":   x0,
                "y":   y0,
                "width":  x1-x0,
                "height":   y1-y0,
                "confidence":  score,
                "name": "rat"
                }
                result ["model_data"]["objects"].append(o_info)

    return json.dumps(result, indent=4)

# if __name__ == "__main__":
#     img_path = "/home/data/888/176.jpg"
#     img = cv2.imread(img_path)
#     model = init()
#     res = process_image(handle=model, input_image=img)
#     print(res)
