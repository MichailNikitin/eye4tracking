# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size,
                           non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import select_device, time_sync

from utils.augmentations import letterbox

weights = 'barhuman_yolov5.pt'
commands = ["center", "right", "left"]
com_num = 0


@torch.no_grad()
def inicial(
        weights,
        data=r'\data\crowdhuman.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    global model, stride, names, pt, jit, onnx, engine, devices

    # Load model
    devices = select_device(device)
    model = DetectMultiBackend(weights, device=devices, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    #

    # Half
    if pt or jit:
        model.model.half() if half else model.model.float()


@torch.no_grad()
def predict(img):
    global stride, pt
    global obj_x

    augment = False,  # augmented inference
    imgsz = [128, 128]
    max_det = 1

    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    im = letterbox(img, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im0s = img
    t1 = time_sync()
    im = torch.from_numpy(im).to(devices)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1
    s = ''

    # Inference
    pred = model(im, augment=augment, visualize=False)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, 0.60, 0.80, None, False, max_det=max_det)
    dt[2] += time_sync() - t3

    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        im0 = im0s.copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                obj_x = xywh[0]
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        if abs(obj_x - 0.5) < 0.05:
            com_num = 0
        elif obj_x > 0.5:
            com_num = 1
        else:
            com_num = 2
        print(f'{s} it takes {t3 - t2:.3f}')
        return com_num


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    frame = False
    while True:
        _, img = cam.read()
        if not frame:
            inicial(weights=weights)
            frame = True
        else:
            com_num = predict(img)
            print(commands[com_num])
