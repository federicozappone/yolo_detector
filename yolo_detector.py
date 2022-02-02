import cv2
import torch
import numpy as np
import time
import torch.backends.cudnn as cudnn

from utils.datasets import letterbox
from utils.general import (check_img_size, non_max_suppression, apply_classifier, 
                           scale_coords, xyxy2xywh, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *


class YOLO_Detector():

    def __init__(self, cfg, weight, imgsz):
        self.cfg = cfg
        self.weights = weight
        self.imgsz = imgsz

        # initialize and select device
        self.device = select_device()
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA

        # load model
        self.model = Darknet(self.cfg, self.imgsz).cuda()
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])

        # move to device
        self.model.to(self.device).eval()

        if self.half:
            self.model.half()  # to FP16

        cudnn.benchmark = True  # set True to speed up constant image size inference

    def detect(self, img0):
        im0 = img0.copy()

        # padded resize
        img = letterbox(img0, new_shape=self.imgsz)[0]

        # convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # run inference
        pred = self.model(img, augment=False)[0]

        # apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, agnostic=True)

        det = pred[0]

        if det is not None and len(det):
            # rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        return det
