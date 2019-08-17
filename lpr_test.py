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


def transform_tensor(input_tensor):
    padded_tensor, _ = pad_to_square(input_tensor, 0)
    resize_tensor = resize(padded_tensor, 416)
    output_tensor = resize_tensor.unsqueeze(0)
    ouput_tensor = Variable(output_tensor.type(torch.FloatTensor))
    return output_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="video/seq01.mp4", type=str)


    # License Plate Detection
    parser.add_argument("--plate_config", default="config/plate_obj_tiny.cfg", type=str)
    parser.add_argument("--plate_weights", default="weights/plate_obj_tiny.weights", type=str)
    parser.add_argument("--plate_names", default="data/plate_obj_tiny.names", type=str)
    parser.add_argument("--plate_thres", default=0.5, type=float)
    parser.add_argument("--plate_nms", default=0.4, type=float)
    parser.add_argument("--plate_size", default=416, type=int)
    
    # Character Detection
    parser.add_argument("--char_config", default="config/char_obj_tiny.cfg", type=str)
    parser.add_argument("--char_weights", default="weights/char_obj_tiny.weights", type=str)
    parser.add_argument("--char_names", default="data/char_obj_tiny.names", type=str)
    parser.add_argument("--char_thres", default=0.5, type=float)
    parser.add_argument("--char_nms", default=0.4, type=float)
    parser.add_argument("--char_size", default=416, type=int)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--n_cpu", default=0, type=int)
    opt=parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up my model
    plateModel = Darknet(opt.plate_config, img_size=opt.plate_size).to(device)
    charModel = Darknet(opt.char_config, img_size=opt.char_size).to(device)

    if opt.plate_weights.endswith(".weights"):
        # Load darknet weights
        plateModel.load_darknet_weights(opt.plate_weights)
        charModel.load_darknet_weights(opt.char_weights)
    else:
        # Load checkpoint wieghts
        print("to do...")
    
    # My model to eval
    plateModel.eval()
    charModel.eval()

    ###########
    # Dataloader
    ###########
    
    # load obj names
    p_names = load_classes(opt.plate_names)
    c_names = load_classes(opt.char_names)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Read Video
    cap = cv2.VideoCapture(opt.video_path)
    if cap.isOpened():
        print("Success read video...")

    while True:
        ret, frame = cap.read()
        if ret:
            # Plate detection
            #resize_frame = cv2.resize(frame, (opt.plate_size, opt.plate_size))
            #resize_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
            cvt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cvt_img)
            img_tensor = transforms.ToTensor()(pil_img)
            plate_tensor = transform_tensor(img_tensor)

            with torch.no_grad():
                plate_detections = plateModel(plate_tensor)
                plate_detections = non_max_suppression(plate_detections, opt.plate_thres, opt.plate_nms)
            
            if plate_detections[0] is not None:
                plate_detections = plate_detections[0]

                # rescale box to origin image
                plate_detections = torch.Tensor(plate_detections)
                plate_detections = rescale_boxes(plate_detections, opt.plate_size, frame.shape[:2])
                unique_labels = plate_detections[:, -1].cpu().unique()
                p_num = 0
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in plate_detections:
                    # print("\t Label: %s, Conf: %.5f"%(p_names[int(cls_pred)], cls_conf.item()))

                    # draw plate box & crop plate image
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    plate_img = cvt_img[y1:y2, x1:x2]
                    plate_pil = Image.fromarray(plate_img)
                    char_tensor = transforms.ToTensor()(plate_pil)
                    char_tensor = transform_tensor(char_tensor)
                    

                    # Character detection
                    # resize_plate = cv2.resize(plate_img, (opt.char_size, opt.char_size))
                    # char_tensor = Variable(resize_plate.type(Tensor))
                    # resize_plate = cv2.cvtColor(resize_plate, cv2.BGR2RGB)
                    # resize_plate = resize_plate / 255

                    # char_tensor = torch.from_numpy(resize_plate)
                    # char_tensor = char_tensor.unsqueeze(0)
                    # char_tensor = char_tensor.permute(0,3,1,2)
                    # char_tensor = Variable(char_tensor.type(torch.FloatTensor))
                    
                    with torch.no_grad():
                        char_detections = charModel(char_tensor)
                        char_detections = non_max_suppression(char_detections,
                                                                opt.char_thres,
                                                                opt.char_nms)
                        if char_detections[0] is not None:
                            char_detections = char_detections[0]
                            char_detections = rescale_boxes(char_detections,
                                                                opt.char_size,
                                                                plate_img.shape[:2])
                            char_labels = char_detections[:, -1].cpu().unique()
                            for cx1, cy1, cx2, cy2, c_conf, c_cls_conf, c_cls_pred in char_detections:
                                plate_img = cv2.rectangle(plate_img,
                                                            (cx1, cy1),
                                                            (cx2, cy2),
                                                            (0,255,0), 2)
                    cv2.imshow("CharResult_"+str(p_num), plate_img)
                    p_num += 1
            cv2.imshow("frame", frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Frame error...")
            break
    
    # Release
    cap.release()
    cv2.destroyAllWindows()

    





























