import os
import cv2
import imutils
import time

def main():
    cap = cv2.VideoCapture("video/seq01.mp4")
    # trackers = cv2.MultiTracker_create()

    if cap.isOpened():
        print("Success read video file")

    frame_num = 0

    frame_info = {}
    trackers_dict = {}

    while True:
        f_start = time.time()

        ret, frame = cap.read()

        if not ret:
            print("frame error...")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            trackers_dict = {key : cv2.TrackerKCF_create() for key in range(3)}
            
            for t in range(3):
                bbox = cv2.selectROI(frame)
                cv2.destroyWindow("ROI selector")

                trackers_dict[t].init(frame, bbox)

                # ret = cv2.TrackerKCF_create().init(frame, bbox)
                # trackers.add(cv2.TrackerKCF_create(), frame, bbox)

        # Update kcf tracker list
        del_boxes = []
        for t_num, t in trackers_dict.items():
            ret, b = t.update(frame)

            x1 = int(b[0])
            y1 = int(b[1])
            x2 = x1 + int(b[2])
            y2 = y1 + int(b[3])
            
            # Drawing
            cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255,0), 2)
            

            # append delete boxes
            if b == (0.0, 0.0, 0.0, 0.0):
                del_boxes.append(t_num)
        
        # delete faild tracking boxes
        for d in del_boxes:
            trackers_dict.pop(d)


        # fps
        f_time = time.time() - f_start
        fps = round((1 / f_time), 2)
        cv2.putText(frame, str(fps) + " fps", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
        cv2.putText(frame, str(frame_num) + " frame", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

        


if __name__ == "__main__":
    main()
