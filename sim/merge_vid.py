import cv2
import numpy as np 
import os


img_size = 500
num_step = 2000

width = 2*img_size+5
height = img_size

for seed_val in [10, 11, 12, 17]:
    video_writer = cv2.VideoWriter(
        f'video/merge_seed_{seed_val}.mp4', 
        cv2.VideoWriter_fourcc(*'MP4V'), 
        30, 
        (width, height))
    
    for i in range(0, 2000, 4):
        print(i)
        img_path_3 = f'image/seed_{seed_val}_angles_3/{i}.png'
        img_path_5 = f'image/seed_{seed_val}_angles_5/{i}.png'
        img_3 = cv2.imread(img_path_3)
        img_5 = cv2.imread(img_path_5)
        board = np.zeros((height, width, 3), dtype=np.uint8) + 255
        board[:, :img_size, :] = img_3
        board[:, -img_size:, :] = img_5
        video_writer.write(board)
    video_writer.release()