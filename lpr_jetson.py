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
from torch.autograd import Variable

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
    parser.add_argument("--video_path", default="video/seq01_compress.mp4", type=str)

    # License Plate Detection
    parser.add_argument("--plate_config", default="config/plate-tiny.cfg", type=str)
    parser.add_argument("--plate_weights", default="weights/plate-tiny_4000.weights", type=str)
    parser.add_argument("--plate_names", default="data/plate_obj_tiny.names", type=str)
    parser.add_argument("--plate_thres", default=0.5, type=float)
    parser.add_argument("--plate_nms", default=0.5, type=float)
    parser.add_argument("--plate_size", default=416, type=int)
    
    # Character Detection
    parser.add_argument("--char_config", default="config/char_obj_tiny.cfg", type=str)
    parser.add_argument("--char_weights", default="weights/char-tiny_best.weights", type=str)
    parser.add_argument("--char_names", default="data/char_obj_tiny.names", type=str)
    parser.add_argument("--char_thres", default=0.5, type=float)
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

    # Check Init.
    print("\t => Use ", device)
    print("\t => Plate Information")
    print("\t\t => config : ", opt.plate_config)
    print("\t\t => weights : ", opt.plate_weights)
    print("\t\t => names : ", opt.plate_names)
    print("\t\t => image size : ", opt.plate_size)
    print("\t\t => threshold : ", opt.plate_thres)
    print("\t\t => nms : ", opt.plate_nms)
    print("\t Character Information")
    print("\t\t => config : ", opt.char_config)
    print("\t\t => weights : ", opt.char_weights)
    print("\t\t => names : ", opt.char_names)
    print("\t\t => image size : ", opt.char_size)
    print("\t\t => threshold : ", opt.char_thres)
    print("\t\t => nms : ", opt.char_nms)

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

    # load obj names
    p_names = load_classes(opt.plate_names)
    c_names = load_classes(opt.char_names)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Read Video
    cap = cv2.VideoCapture(opt.video_path)
    if cap.isOpened():
        print("Success read video...")
    
    frame_num = 0
    plate_time_list = []
    char_time_list = []
    while True:
        ret, frame = cap.read()
        if ret:
            f_start = time.time()
            # Plate detection
            cvt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(cvt_img, cv2.COLOR_RGB2GRAY)
            cvt_img =cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

            pil_img = Image.fromarray(cvt_img)
            ## not torchvision
            img_tensor = np.array(pil_img)
            img_tensor = torch.from_numpy(img_tensor).float().to(device)
            img_tensor = img_tensor.permute(2,0,1) / 255.
            plate_tensor = transform_tensor(img_tensor, opt.plate_size, device)

            # for Visualization
            char_detect_size = 0
            result_char = ""

            with torch.no_grad():
                # License Plate Inference Time
                start = time.time() 
                
                plate_detections = plateModel(plate_tensor)
                plate_detections = non_max_suppression(plate_detections, opt.plate_thres, opt.plate_nms)

                plate_time = time.time() - start
                plate_time_list.append(plate_time)
            
            if plate_detections[0] is not None:
                plate_detections = plate_detections[0]
                # rescale box to origin image
                plate_detections = torch.Tensor(plate_detections)
                plate_detections = rescale_boxes(plate_detections, opt.plate_size, cvt_img.shape[:2])
                unique_labels = plate_detections[:, -1].cpu().unique()
                p_num = 0
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in plate_detections:
                    x1 = int(x1.item())
                    y1 = int(y1.item())
                    x2 = int(x2.item())
                    y2 = int(y2.item())

                    # draw plate box & crop plate image
                    cvt_img = cv2.rectangle(cvt_img, (x1, y1), (x2, y2), (0,0,255), 2)
                    plate_img = cvt_img[y1:y2, x1:x2]
                    plate_pil = Image.fromarray(plate_img)

                    ## not torchvision
                    char_tensor = np.array(plate_pil)
                    char_tensor = torch.from_numpy(char_tensor).float().to(device)
                    char_tensor = char_tensor.permute(2,0,1) / 255.
                    
                    char_tensor = transform_tensor(char_tensor, opt.char_size, device)
                    

                    # Character detection
                    with torch.no_grad():
                        c_start = time.time()

                        char_detections = charModel(char_tensor)
                        char_detections = non_max_suppression(char_detections,
                                                                opt.char_thres,
                                                                opt.char_nms)
                        # Character Inference Time.
                        char_time = time.time() - c_start
                        char_time_list.append(char_time)
                        # print("=> char recog time : ", char_time)
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
                    p_num += 1

            f_time = time.time() - f_start
            fps = round((1 / f_time), 2)
            
            # Put text
            cv2.putText(cvt_img, str(fps) + " fps", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            if char_detect_size > 5:
                cv2.putText(cvt_img, result_char, (200, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                print("\t ===> Character : ", result_char)

            h,w = cvt_img.shape[:2]
            cv2.imshow("convert frame", cvt_img)        

            frame_num += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Frame error...")
            break
    
    # Release
    cap.release()
    cv2.destroyAllWindows()

    # Check average inference time
    print("\n\t==>LPR Inference Time")
    print("\t\t==>Plate Detection : ", str(sum(plate_time_list) / len(plate_time_list)))
    print("\t\t==>Character Recognition : ", str(sum(char_time_list) / len(char_time_list)))
