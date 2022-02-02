#!/usr/bin/env python

import cv2
import numpy as np
import torch

from yolo_detector import YOLO_Detector


def draw_detection(x1, y1, x2, y2, id, c, frame, color=(255, 0, 0), font_thickness=1, font_size=1, labels=None):

    pt1 = int(x1), int(y1)
    pt2 = int(x2), int(y2)

    cv2.rectangle(frame, pt1, pt2, color, font_thickness)

    class_id = int(id.item())

    if labels is None:
        label = str(class_id)
    else:
        label = labels[class_id]

    text_size = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)

    center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
    pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
        text_size[0][1]

    cv2.rectangle(frame, pt1, pt2, color, -1)
    cv2.putText(frame, label, center, cv2.FONT_HERSHEY_PLAIN,
                font_size, (255, 255, 255), font_thickness)

      
def main():
    labels_path = "data/coco.names"
    
    detector_cfg_path = "cfg/yolor_p6.cfg"
    detector_weights_path = "weights/yolor_p6.pt"

    # read class labels from file
    with open(labels_path) as file:
        labels = [line.rstrip() for line in file.readlines()]

    # initialize detector
    with torch.no_grad():
        detector = YOLO_Detector(detector_cfg_path, detector_weights_path, 640)


    # load image with opencv
    image = cv2.imread("image.png")

    detections = detector.detect(image)

    xyxy = detections.cpu()[:, 0:4]
    confs = detections.cpu()[:, 4]
    classes = detections.cpu()[:, 5]

    # draw detections on image
    for (x1, y1, x2, y2), id, confidence in zip(xyxy, classes, confs):
        # adjust font tickness and size for the best result on your monitor
        draw_detection(x1, y1, x2, y2, id, confidence, image, font_thickness=1, font_size=1, labels=labels)

    # write results on file
    cv2.imwrite("detections.png", image)

    # display results
    cv2.imshow("detections", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
