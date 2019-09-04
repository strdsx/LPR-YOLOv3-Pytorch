import cv2
import os

if __name__ == "__main__":
    img_folder = "car/"
    img_list = os.listdir(img_folder)
    for i in img_list:
        img = cv2.imread(img_folder + i, 0)
        cv2.imwrite(img_folder + i, img)
        img = cv2.imread(img_folder + i, cv2.IMREAD_COLOR)
        cv2.imwrite(img_folder + i, img)
