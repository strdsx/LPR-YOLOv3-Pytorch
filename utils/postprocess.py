import cv2
import numpy as np

def get_name(object_id):
    kr_names = ["0","1","2","3","4","5","6","7","8","9",
        "가","나","고","노","다","라","마","거","너","더","러","머","도","로","모",
        "구","누","두","루","무","버","서","어","저","보","소","오","조","부","수","우",
        "주","허","바","사","아","자","배","하","호",
        "부산","대구","인천","경기","대전","울산","경기","강원","충북","충남","전북","전남","경북","경남","제주","서울","세종",
        "부산","대구","인천","경기","대전","울산","경기","강원","충북","충남","전북","전남","경북","경남","제주","서울","세종"
    ]
    char_type = "NUMBER"
    if 9 < object_id < 50:
        char_type = "SINGLE"
    elif object_id > 49:
        char_type = "AREA"
    
    return kr_names[object_id], char_type

def delete_overlap(input_array):
    error_boxes = []
    for i, info in enumerate(input_array):
        if i < (len(input_array) -1):
            # Current box info.
            curr_bw = info[2] - info[0]
            curr_cx = info[0] + float(curr_bw / 2)

            # Next box info.
            next_x1 = input_array[i+1][0]

            # Find overlap box
            if next_x1 < curr_cx:
                # Compare Class Confidence
                curr_cls_conf = info[4]
                next_cls_conf = input_array[i+1][4]
                if curr_cls_conf > next_cls_conf:
                    error_index = i+1
                else:
                    error_index = i
                    
                error_boxes.append(error_index)

    # Delete overlap boxes
    for e in error_boxes:
        del input_array[e]

    return input_array

'''
def delete_errorbox(input_array):
    error_boxes = []
    
    if len(input_array) > 6: # max number of character
        # Thin plate
'''

def sort_boxes(char_detections):
    # x1, y1, x2, y2, prob, cls_conf, object_id
    '''
    plate_tensor = np.array(
        [[56.780655, 8.455865, 63.07199,18.356714, 0.9345888, 0.9963361, 5.],
        [50.26172, 8.582612, 56.638973, 18.846937, 0.93030804, 0.99982554, 0.],
        [62.7273, 7.9615235, 69.48192, 17.991077, 0.9639955, 0.9479099, 9.],
        [69.439125, 7.4344854, 75.984375, 17.992281, 0.9599736, 0.89220864, 9.],
        [33.128117, 9.394947, 39.34961, 19.837324, 0.8056386, 0.9879038, 6.],
        [26.695444, 8.803099, 33.28907, 19.630323, 0.5839797, 0.9987348, 3.],
        [39.130512, 8.719946, 48.26919, 19.141325, 0.85395133, 0.63955206, 14.]]
        )
    '''

    # Y sort
    y_sorted = sorted(char_detections, key=lambda y_value: y_value[1])
    bottom_line = []
    top_line = [y_sorted[0]]

    for i in enumerate(y_sorted):       
        if i[0] < (len(y_sorted) - 1):
            idx = i[0] # box number
            arr = i[1] # array info
            
            curr_height = arr[3] - arr[1]
            pivot_height = arr[1] + float(curr_height / 2)

            # find bottom line
            if y_sorted[idx+1][1] > pivot_height:
                bottom_line.append(y_sorted[idx+1])
            else:
                top_line.append(y_sorted[idx+1])

    # X sort
    top_x_sorted = sorted(top_line, key=lambda x_value: x_value[0])
    bottom_x_sorted = sorted(bottom_line, key=lambda x_value: x_value[0])

    # Delete overlap box
    top_x_sorted = delete_overlap(top_x_sorted)
    bottom_x_sorted = delete_overlap(bottom_x_sorted)

    # Final box 
    final_boxes = top_x_sorted
    if len(bottom_x_sorted) > 0:
        final_boxes = top_x_sorted.extend(bottom_x_sorted)

    return final_boxes        