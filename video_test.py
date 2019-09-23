import os
import cv2

def main():
    cap = cv2.VideoCapture("video/seq01.mp4")
    if cap.isOpened():
        print("Success read video file")

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("frame error...")


if __name__ == "__main__":
    main()
