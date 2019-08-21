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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2

def main():
    device = "cpu"
    charModel = Darknet("config/char_obj_tiny.cfg", 416).to(device)
    charModel.load_darknet_weights("weights/char_obj_tiny.weights")
    charModel.eval()
    
    char_names = load_classes("data/char_obj_tiny.names")

    loader = transforms.Compose([transforms.Scale((416,416)), transforms.ToTensor()])

    img_path = "./data/plate/plate.jpg"
    img = cv2.imread(img_path)
    cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize_img = cv2.resize(cvt_img, (416,416))
    pil_img = Image.fromarray(cvt_img)
    char_tensor = transforms.ToTensor()(pil_img)
    char_tensor, _ = pad_to_square(char_tensor, 0)
    char_tensor = resize(char_tensor, 416)
    char_tensor = char_tensor.unsqueeze(0)
    char_tensor = char_tensor

    char_tensor = Variable(char_tensor.type(torch.FloatTensor))
    print(char_tensor)
    
    with torch.no_grad():
        char_detections = charModel(char_tensor)
        char_detections = non_max_suppression(char_detections, 0.8, 0.4)
    
    if char_detections[0] is not None:
        char_detections = char_detections[0]
        char_detections = torch.Tensor(char_detections)
        char_detections = rescale_boxes(char_detections, 416, cvt_img.shape[:2])
        unique = labels = char_detections[:,-1].cpu().unique()
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in char_detections:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow("plate", img)
        cv2.waitKey()
        cv2.destroyAllWindows()















if __name__ == "__main__":
    main()
