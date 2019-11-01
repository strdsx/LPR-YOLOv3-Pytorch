from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.postprocess import *

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
import time

import imutils

# Transform image tensor (PIL), return final image tensor
def transform_tensor(input_tensor, image_size, current_device):
    padded_tensor, _ = pad_to_square(input_tensor, 0)
    resize_tensor = resize(padded_tensor, image_size)
    output_tensor = resize_tensor.unsqueeze(0)

    ouput_tensor = Variable(output_tensor.type(torch.FloatTensor))
    output_tensor = output_tensor.to(current_device)
    return output_tensor

# def update_trackers(capture, trackers_dict):
#     _, curr_frame = cpature.read()
#     if len(trackers_dict) > 0:
#         for p_id, track in trackers_dict.items():
#             ok, tb = track.update(frame)

#             if tb == (0.0, 0.0, 0.0, 0.0):
#                         del_boxes.append(p_id)
#                     else:
#                         px1 = int(tb[0])
#                         py1 = int(tb[1])
#                         px2 = px1 + int(tb[2])
#                         py2 = py1 + int(tb[3])

#                         cv2.putText(draw_frame, "Tracking", (px1, py2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
#                         cv2.rectangle(draw_frame,(px1 - 5 , py1 - 5), (px2 + 5 , py2 + 5), (255,255,0), 2)

#                 # Delete Tracking Fail boxes
#                 for d in del_boxes:
#                     trackers_dict.pop(d)

#     return trackers_dict
    

def PlateDetection(plate_tensor, plateModel):
    with torch.no_grad():
        start = time.time()

        plate_detections = plateModel(plate_tensor)
        plate_detections = non_max_suppression(plate_detections, opt.plate_thres, opt.plate_nms)

        plate_time = float(time.time() - start) * 1000

        if plate_detections[0] is not None:
            plate_detections = plate_detections[0]
            # rescale box to origin image
            plate_detections = torch.Tensor(plate_detections)
            plate_detections = rescale_boxes(plate_detections, opt.plate_size, cvt_img.shape[:2])
        else:
            plate_detections = []

    return plate_detections, plate_time



def CharRecognition(input_image, color_id, charModel):
    with torch.no_grad():
        plate_pil = Image.fromarray(input_image)

        # to Tensor
        char_tensor = transforms.ToTensor()(plate_pil)
        char_tensor = transform_tensor(char_tensor, opt.char_size, device)

        c_start = time.time()
        char_detections = charModel(char_tensor)
        char_detections = non_max_suppression(char_detections,
                                                opt.char_thres,
                                                opt.char_nms)

        char_time = float(time.time() - c_start) * 1000

        # Error prevention
        if char_detections[0] is not None:
            char_detections = char_detections[0]
            char_detections = rescale_boxes(char_detections,
                                                opt.char_size,
                                                input_image.shape[:2])

            # Postprocessing
            sorted_boxes = sort_boxes(char_detections)
            sorted_boxes = char_condition(color_id, sorted_boxes)
        else:
            sorted_boxes = []
    
    return sorted_boxes, char_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="video/seq01.mp4", type=str)

    # License Plate Detection
    parser.add_argument("--plate_config", default="config/color-plate.cfg", type=str)
    parser.add_argument("--plate_weights", default="weights/plate_train_last.weights", type=str)
    parser.add_argument("--plate_names", default="data/color-plate.names", type=str)
    parser.add_argument("--plate_thres", default=0.5, type=float)
    parser.add_argument("--plate_nms", default=0.5, type=float)
    parser.add_argument("--plate_size", default=416, type=int)

    # Character Detection
    parser.add_argument("--char_config", default="config/pchar-tiny.cfg", type=str)
    parser.add_argument("--char_weights", default="weights/pchar-tiny_best.weights", type=str)
    parser.add_argument("--char_names", default="data/pchar84.names", type=str)
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
    print("\t => Device ", device)

    print("\t => Plate Information")
    print("\t\t => config : ", opt.plate_config)
    print("\t\t => weights : ", opt.plate_weights)
    print("\t\t => names : ", opt.plate_names)
    print("\t\t => image size : ", opt.plate_size)
    print("\t\t => threshold : ", opt.plate_thres)
    print("\t\t => nms : ", opt.plate_nms)

    print("\t => Character Information")
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

    # LPR model to eval
    plateModel.eval()
    charModel.eval()

    # load obj names (Char : EN name)
    p_names = load_classes(opt.plate_names)
    c_names = load_classes(opt.char_names)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Read Video
    cap = cv2.VideoCapture(opt.video_path)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter("output_video/lpr_output.mp4", fourcc, 30.0, (640, 480))
    if cap.isOpened():
        print("Success read video...")

    # Time list
    plate_time_list = []
    char_time_list = []

    # for video post-processing
    plate_list = []
    char_list = []
    result_char = ""

    # Tracker
    trackers_dict = {}
    tb = None

    frame_num = 0

    while True:
        # prev_frame = frame[:]
        ret, frame = cap.read()
        draw_frame = frame.copy()
        if ret:
            f_start = time.time()

            # Plate detection
            # half_frame = frame[:int(frame.shape[0]/2), :]
            cvt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(cvt_img, cv2.COLOR_RGB2GRAY)
            cvt_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

            pil_img = Image.fromarray(cvt_img)

            # torchvision
            img_tensor = transforms.ToTensor()(pil_img)

            # YOLOv3 Area
            # frame = cv2.line(frame, (0, int(frame.shape[0]/3)), (int(frame.shape[1]),int(frame.shape[0]/3)), (0,0,255), 2)

            ## not torchvision
            '''
            img_tensor = np.array(pil_img)
            img_tensor = torch.from_numpy(img_tensor).float().to(device)
            img_tensor = img_tensor.permute(2,0,1) / 255.
            '''

            plate_tensor = transform_tensor(img_tensor, opt.plate_size, device)

            plate_detections, plate_time = PlateDetection(plate_tensor, plateModel)
            plate_time_list.append(plate_time)


            ######################
            ### Update Tracker ###
            ######################
            del_boxes = []
            if len(trackers_dict) > 0:
                for p_id, track in trackers_dict.items():

                    ret, tb = track.update(frame)

                    if tb == (0.0, 0.0, 0.0, 0.0):
                        del_boxes.append(p_id)
                    else:
                        px1 = int(tb[0])
                        py1 = int(tb[1])
                        px2 = px1 + int(tb[2])
                        py2 = py1 + int(tb[3])
                        
                        cv2.putText(draw_frame, "Tracking", (px1, py2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                        cv2.rectangle(draw_frame,(px1 - 5 , py1 - 5), (px2 + 5 , py2 + 5), (255,255,0), 2)

                # Delete Tracking Fail boxes
                for d in del_boxes:
                    trackers_dict.pop(d)
                
            # Tracker dict
            if len(plate_detections) > 0:
                # trackers_dict = {key : cv2.TrackerKCF_create() for key in range(len(plate_detections))}
                for plate_id, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(plate_detections):
                    # YOLO Box
                    x1 = int(x1.item())
                    y1 = int(y1.item())
                    x2 = int(x2.item())
                    y2 = int(y2.item())

                    # Plate ID, Plate Color
                    plate_info = str(plate_id) + "_" + str(int(cls_pred.cpu()))
                    
                    # Tracker Init
                    trackers_dict[plate_info] = cv2.TrackerKCF_create()
                    trackers_dict[plate_info].init(frame, (x1, y1, x2-x1, y2-y1))
            else:
                # If Plate Detection Fail => Update Tracker
                del_boxes = []
                if len(trackers_dict) > 0:
                    for p_id, track in trackers_dict.items():

                        ret, tb = track.update(frame)

                        if tb == (0.0, 0.0, 0.0, 0.0):
                            del_boxes.append(p_id)
                        else:
                            # Plate box
                            px1 = int(tb[0])
                            py1 = int(tb[1])
                            px2 = px1 + int(tb[2])
                            py2 = py1 + int(tb[3])

                            # cv2.putText(draw_frame, "Tracking", (px1, py2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                            # cv2.rectangle(draw_frame,(px1 - 5 , py1 - 5), (px2 + 5 , py2 + 5), (255,255,0), 2)

                            # Crop Tracking Box
                            plate_img = cvt_img[py1:py2, px1:px2]

                            # Character Recognition Start
                            color_index = int(p_id.split("_")[1])
                            char_boxes, char_time = CharRecognition(plate_img, color_index, charModel)
                            char_time_list.append(char_time)

                            # Result of License Plate Character recognitions
                            tracking_char = ""
                            for cx1, cy1, cx2, cy2, c_conf, c_cls_conf, c_cls_pred in char_boxes:
                                # License plate char result
                                pred_index = int(c_cls_pred.cpu())

                                ## Plate Char EN
                                # result_char += c_names[pred_index]

                                # Plate Char KR
                                # get_char, _  = get_name(pred_index)
                                # result_char += get_char
                                tracking_char += get_name(pred_index)[0]

                                # Draw character detection boxes
                                draw_frame = cv2.rectangle(draw_frame, (x1 + cx1, y1 + cy1), (x1 + cx2, y1 + cy2), (255,255,0), 1)
                            print("\tTracking Reuslt ==> ", tracking_char)

                    # Delete Tracking Fail boxes
                    for d in del_boxes:
                        trackers_dict.pop(d)

            # Result of plate detections
            for plate_id, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(plate_detections):
                color_id = int(cls_pred.cpu())
                plate_color = p_names[color_id]

                # Plate color (white, yellow, green)
                if color_id == 0:
                    draw_color = (255, 255, 255)
                elif color_id == 1:
                    draw_color = (0, 255, 255)
                elif color_id == 2:
                    draw_color = (0, 255, 0)

                # YOLOv3 result of plate detection
                x1 = int(x1.item())
                y1 = int(y1.item())
                x2 = int(x2.item())
                y2 = int(y2.item())

                # Add Tracking List
                trackers_dict[str(plate_id) + "_" + str(int(cls_pred.cpu()))] = cv2.TrackerKCF_create()
                trackers_dict[str(plate_id) + "_" + str(int(cls_pred.cpu()))].init(frame, (x1, y1, x2 - x1, y2 - y1))

                # draw yolo plate box (yolov3 result)
                draw_frame = cv2.rectangle(draw_frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(draw_frame, plate_color, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

                # Error prevention
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > cvt_img.shape[1]:
                    x2 = cvt_img.shape[1]
                if y2 > cvt_img.shape[0]:
                    y = cvt_img.shape[0]

                plate_img = cvt_img[y1:y2, x1:x2]

                # Character Recognitions
                char_boxes, char_time = CharRecognition(plate_img, color_id, charModel)
                char_time_list.append(char_time)

                # Result of License Plate Character recognitions
                yolo_char = ""
                for cx1, cy1, cx2, cy2, c_conf, c_cls_conf, c_cls_pred in char_boxes:
                    # License plate char result
                    pred_index = int(c_cls_pred.cpu())

                    # # Plate Char EN
                    # result_char += c_names[pred_index]

                    # Plate Char KR
                    # get_char, _  = get_name(pred_index)
                    # result_char += get_char
                    yolo_char += get_name(pred_index)[0]

                    # Draw character detection boxes
                    draw_frame = cv2.rectangle(draw_frame, (x1 + cx1, y1 + cy1), (x1 + cx2, y1 + cy2), (255, 255, 0), 2)
                print("\tResult ==> ", yolo_char)
                # cv2.putText(draw_frame, str(yolo_char), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # # Result of Character
            # if len(char_detections) > (min_char_length(color_id) - 1):
            #     print("Frame => {} \tColor => {} \tChar => {} \tPlate Inf Time => {}ms \tChar Inf Time => {}ms".format(
            #         frame_num, plate_color, result_char, round(plate_time, 2) ,round(char_time, 2))
            #         )

            # FPS
            f_time = time.time() - f_start
            fps = round((1 / f_time), 2)
            cv2.putText(frame, str(fps) + " fps", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # cv2.namedWindow("frame")

            # cv2.moveWindow("frame", 3840,500)
            draw_frame = cv2.resize(draw_frame, None, fx=2, fy=2)
            cv2.imshow("frame", draw_frame)
            writer.write(frame)

            # cv2.imwrite("frame/" + str(img) + ".jpg", draw_frame)

            frame_num += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Frame error...")
            break

    # Release
    cap.release()
    cv2.destroyAllWindows()

    # Average inference time
    print("\n\t==>LPR Inference Time")
    print("\t\t==>Plate Detection : ", str(sum(plate_time_list) / len(plate_time_list)))
    print("\t\t==>Character Recognition : ", str(sum(char_time_list) / len(char_time_list)))
