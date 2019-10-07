import os
import cv2
import imutils
import time

def main():
    cap = cv2.VideoCapture("video/seq01.mp4")
    tracker = cv2.TrackerKCF_create()

    if cap.isOpened():
        print("Success read video file")
    bbox = None
    frame_num = 0

    while True:
        f_start = time.time()

        ret, frame = cap.read()

        if not ret:
            print("frame error...")
            break

        if cv2.waitKey(1) & 0xFF == ord('s'):
            bbox = cv2.selectROI(frame)
            tracker = cv2.TrackerKCF_create()
            cv2.destroyWindow("ROI selector")

            ret = tracker.init(frame, bbox)

            ret, bbox = tracker.update(frame)
        
        ret, bbox = tracker.update(frame)

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = x1 + int(bbox[2])
        y2 = y1 + int(bbox[3])
        cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255,0), 2)

        # fps
        f_time = time.time() - f_start
        fps = round((1 / f_time), 2)
        cv2.putText(frame, str(fps) + " fps", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        


if __name__ == "__main__":
    main()
