import os
import numpy as np
import cv2

# generate triplet image nparrray for training from videos
width = 416
height = 128
every_n = 3
video_list = os.listdir('video')
for video_name in video_list:
    count = 0
    video_path = f'video/{video_name}'
    print(f'Processing {video_path}')
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    triplet = []
    while(cap.isOpened()):
        for i in range(every_n):
            ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            triplet.append(frame)
            if len(triplet) == 3:
                triplet_array = np.zeros((height, width*3, 3))
                for i in range(3):
                    triplet_array[:, width*i:width*(i+1), :] = triplet[i]
                base_name = f'{video_name}_{count}'
                cv2.imwrite(f'train/{base_name}.png', triplet_array)
                count += 1
                del triplet[0]
        else: 
            break
    cap.release()