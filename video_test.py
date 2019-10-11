from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import cv2

# Transform image tensor (PIL), return final image tensor
def transform_tensor(input_tensor, image_size, current_device):
    padded_tensor, _ = pad_to_square(input_tensor, 0)
    resize_tensor = resize(padded_tensor, image_size)
    output_tensor = resize_tensor.unsqueeze(0)

    ouput_tensor = Variable(output_tensor.type(torch.FloatTensor))
    output_tensor = output_tensor.to(current_device)
    return output_tensor



def main():
    input_image_size = 416
    device = "cuda"
    Model = Darknet("config/yolov3.cfg", input_image_size).to(device)
    Model.load_darknet_weights("weights/yolov3.weights")
    Model.eval()
    
    char_names = load_classes("data/coco.names")

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # img_path = "./data/plate/plate.jpg"
    # img = cv2.imread(img_path)

    cap = cv2.VideoCapture("video/191011_driving.mp4")
    if cap.isOpened():
        print("Success read video...")

    frame_number = 0
    while True:
        f_start = time.time()

        ret, frame = cap.read()
        if ret is None: break

        cvt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize_img = cv2.resize(cvt_img, (416,416))
        pil_img = Image.fromarray(cvt_img)
        char_tensor = transforms.ToTensor()(pil_img)

        char_tensor = transform_tensor(char_tensor, input_image_size, device)
        
        with torch.no_grad():
            char_detections = Model(char_tensor)
            char_detections = non_max_suppression(char_detections, 0.8, 0.4)
        
            if char_detections[0] is not None:
                char_detections = char_detections[0]
                char_detections = torch.Tensor(char_detections)
                char_detections = rescale_boxes(char_detections, input_image_size, cvt_img.shape[:2])
                unique = labels = char_detections[:,-1].cpu().unique()
                # Drawing
                
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in char_detections:
                    name = int(cls_pred.item())
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, char_names[name], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        
        f_time = time.time() - f_start
        fps = round((1 / f_time), 2)

        cv2.putText(frame, str(fps) + " fps", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.imshow("plate", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1


if __name__ == "__main__":
    main()
