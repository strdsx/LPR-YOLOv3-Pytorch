from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import numpy as np

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
# from torchvision import datasets
from torch.autograd import Variable

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.ticker import NullLocator
import cv2
import time

# Transform image tensor (PIL), return final image tensor
def transform_tensor(input_tensor, image_size, current_device):
    padded_tensor, _ = pad_to_square(input_tensor, 0)
    resize_tensor = resize(padded_tensor, image_size)
    output_tensor = resize_tensor.unsqueeze(0)

    ouput_tensor = Variable(output_tensor.type(torch.FloatTensor))
    output_tensor = output_tensor.to(current_device)
    return output_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="image/char", type=str)

    # License Plate Detection
    parser.add_argument("--plate_config", default="config/plate-tiny.cfg", type=str)
    parser.add_argument("--plate_weights", default="weights/plate-tiny.weights", type=str)
    parser.add_argument("--plate_names", default="data/plate_obj_tiny.names", type=str)
    parser.add_argument("--plate_thres", default=0.5, type=float)
    parser.add_argument("--plate_nms", default=0.5, type=float)
    parser.add_argument("--plate_size", default=416, type=int)
    
    # Character Detection
    parser.add_argument("--char_config", default="config/char_obj_tiny.cfg", type=str)
    parser.add_argument("--char_weights", default="weights/pchar-tiny.weights", type=str)
    parser.add_argument("--char_names", default="data/pchar84.names", type=str)
    parser.add_argument("--char_thres", default=0.6, type=float)
    parser.add_argument("--char_nms", default=0.5, type=float)
    parser.add_argument("--char_size", default=416, type=int)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--n_cpu", default=0, type=int)
    parser.add_argument("--cuda", default="cuda", type=str, help="cpu or cuda")
    opt=parser.parse_args()
    
    if opt.cuda == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    plateModel = Darknet(opt.plate_config, img_size=opt.plate_size).to(device)
    charModel = Darknet(opt.char_config, img_size=opt.char_size).to(device)

    if opt.char_weights.endswith(".weights"):
        # Load darknet weights
        plateModel.load_darknet_weights(opt.plate_weights)
        charModel.load_darknet_weights(opt.char_weights)

    else:
        print("\n\t ===> Read Model Fail...")

    plateModel.eval()
    charModel.eval()

    # load obj names
    p_names = load_classes(opt.plate_names)
    c_names = load_classes(opt.char_names)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # input image list
    img_list = [x for x in os.listdir(opt.image_path) if x.split(".")[1] == "jpg"]

    for i in img_list:
        plate_img = cv2.imread(opt.image_path + "/" + i)
        rgb_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        plate_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(plate_img)
        # img_tensor = transforms.ToTensor()(pil_img)

        # jetson
        img_tensor = np.array(pil_img)
        img_tensor = torch.from_numpy(img_tensor).float().to(device)
        img_tensor = img_tensor.permute(2,0,1) / 255.

        plate_tensor = transform_tensor(img_tensor, opt.plate_size, device)

        # for Visualization
        char_detect_size = 0
        result_char = ""

        with torch.no_grad():
            char_detections = charModel(plate_tensor)
            char_detections = non_max_suppression(char_detections, opt.char_thres, opt.char_nms)

            if char_detections[0] is not None:
                char_detections = char_detections[0]

                char_detections = rescale_boxes(char_detections,
                                                    opt.char_size,
                                                    plate_img.shape[:2])
                char_labels = char_detections[:, -1].cpu().unique()
                char_detect_size = len(char_detections)
                for cx1, cy1, cx2, cy2, c_conf, c_cls_conf, c_cls_pred in char_detections:

                    # License plate char result
                    pred_index = int(c_cls_pred.cpu())
                    result_char += c_names[pred_index]

                    # Draw character detection boxes
                    plate_img = cv2.rectangle(plate_img,
                                                (cx1, cy1),
                                                (cx2, cy2),
                                                (0,255,0), 2)

        cv2.imshow("Character", plate_img)
        cv2.waitKey()
        cv2.destroyAllWindows()














