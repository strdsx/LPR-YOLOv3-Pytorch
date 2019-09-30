import os
import cv2

def main():
    img_folder = "./data/car/"

    img_list = os.listdir(img_folder)

    for i in img_list:
        img_path = img_folder + i

        img = cv2.imread(img_path, 0)

        cv2.imwrite(img_path, img)

        gray_3ch = cv2.imread(img_path)
        cv2.imwrite(img_path, gray_3ch)

    

if __name__ == '__main__':
    main()