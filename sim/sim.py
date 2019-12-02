from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

from motion_planner import MotionPlanner
from util import *


for seed_val in [10, 11, 12, 17]:
    for dirction_val in [3, 5]:
        np.random.seed(seed_val)
        num_directions = dirction_val
        # constants
        img_size = 500
        robot_size = 5
        num_step = 2000
        max_random = 50
        min_obstacle = 4
        max_obstacle = 8
        min_dist = 25
        num_obstacle = np.random.randint(min_obstacle, max_obstacle)

        BACKGROUND_COLOR = (50, 50, 50)
        OBSTACLE_COLOR = (255, 255, 255)
        ROBOT_COLOR = (0, 255, 0)
        DISTANCE_COLOR = (255, 213, 3)
        WARNING_COLOR = (255, 0, 0)
        PATH_COLOR = (252, 3, 236)
            
        # angles arranged in the order of :
        #        0 15 14 13 12 
        #        1          11
        #        2          10
        #        3           9
        #        4  5  6  7  8

        if num_directions == 3:
            num_angles = 8
            angle_vectors = np.array([
                [-2, -2], [-2,  0], [-2,  2], [ 0,  2],
                [ 2,  2], [ 2,  0], [ 2, -2], [ 0, -2]
            ])
            options = [-1, 0, 1]
        elif num_directions == 5:
            num_angles = 16
            angle_vectors = np.array([
                [-2, -2], [-2, -1], [-2,  0], [-2,  1], 
                [-2,  2], [-1,  2], [ 0,  2], [ 1,  2], 
                [ 2,  2], [ 2,  1], [ 2,  0], [ 2, -1], 
                [ 2, -2], [ 1, -2], [ 0, -2], [-1, -2]
            ])
            options = [-2, -1, 0, 1, 2]


        # NOTE: nparray is (row, col), ImageDraw is (col, row)

        def create_obstacles(num_obstacle):
            obstacle_list = []
            for i in range(num_obstacle):
                corners = get_random_rect(img_size)
                obstacle_list.append(corners)
            return obstacle_list


        def draw_obstacles(draw, obstacle_list):
            for corners in obstacle_list:
                shape_tuple = to_tuple(corners)
                draw.polygon(xy=shape_tuple, fill=OBSTACLE_COLOR, outline=OBSTACLE_COLOR)
            return draw


        def get_empty_coords(image, img_size):
            empty_coords = []
            for row in range(img_size):
                for col in range(img_size):
                    if not np.all(image[row, col, :] == np.array(OBSTACLE_COLOR)):
                        empty_coords.append([col, row])
            return empty_coords


        def choose_random_empty_coords(empty_coords):
            robot_coord_choice = np.random.choice(len(empty_coords))
            choice = np.array(empty_coords[robot_coord_choice])
            return choice


        def generate_robot_rect(robot_coord):
            corners = [
                robot_coord - [robot_size, robot_size], 
                robot_coord + [robot_size, -robot_size], 
                robot_coord + [robot_size, robot_size], 
                robot_coord + [-robot_size, robot_size]
                ]
            return corners


        def draw_robot(draw, robot_rect):
            shape_tuple = to_tuple(robot_rect)
            draw.polygon(xy=shape_tuple, fill=ROBOT_COLOR, outline=ROBOT_COLOR)
            return draw


        def get_time_str():
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            return dt_string


        def init_video_writer(dt_string):
            video_writer = cv2.VideoWriter(f'video/{dt_string}.mp4', 
                cv2.VideoWriter_fourcc(*'MP4V'), 
                30, 
                (img_size, img_size))
            return video_writer


        def get_new_board():
            board = Image.new(mode="RGB", size=(img_size, img_size), color=BACKGROUND_COLOR)
            return board


        def get_idx_around_angle(robot_angle):
            idx = robot_angle + np.array(options)
            idx = idx % num_angles
            idx = idx.astype(np.int)
            return idx


        def measure_dists(image, robot_coord, angle_vectors):
            dist_map = []
            end_points = []
            for angle_vector in angle_vectors:
                dist, col_end, row_end = get_dist_at_angle(image, img_size, robot_coord, angle_vector)
                dist_map.append(dist)
                end_points.append([col_end, row_end])
            dist_map = np.array(dist_map)
            return dist_map, end_points


        def draw_dists(draw, robot_coord, end_points, idx):
            [col_robot, row_robot] = robot_coord
            for j in idx:
                col_end, row_end = end_points[j]
                draw.line((col_robot, row_robot, col_end, row_end), fill=DISTANCE_COLOR, width = 3)
                if dist_map[j] <= min_dist:
                    draw.line((col_robot, row_robot, col_end, row_end), fill=WARNING_COLOR, width = 3)
            return draw


        def prep_board_for_video(board):
            image = np.asarray(board)
            image = np.copy(image)
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image


        def choose_random_angle():
            return np.random.randint(num_angles)


        def draw_path(previous_robot_coords, obstacle_list):
            board = get_new_board()
            draw = ImageDraw.Draw(board)
            draw_obstacles(draw, obstacle_list)
            for i in range(len(previous_robot_coords) - 1):
                previous_robot_coord = previous_robot_coords[i]
                robot_coord = previous_robot_coords[i+1]
                [col_robot, row_robot] = robot_coord
                [col_end, row_end] = previous_robot_coord
                draw.line((col_robot, row_robot, col_end, row_end), fill=PATH_COLOR, width = 3)
            image = np.asarray(board)
            image = image.astype(np.uint8)
            cv2.imwrite(f'path/{num_directions}/{num_directions}_{seed_val}.png', image)


        board = get_new_board()
        draw = ImageDraw.Draw(board)
        obstacle_list = create_obstacles(num_obstacle)
        draw_obstacles(draw, obstacle_list)
        image = np.asarray(board)
        empty_coords = get_empty_coords(image, img_size)
        robot_coord = choose_random_empty_coords(empty_coords)
        previous_robot_coord = np.copy(robot_coord)
        robot_rect = generate_robot_rect(robot_coord)
        draw = draw_robot(draw, robot_rect)
        robot_angle = choose_random_angle()
        angle_list = np.linspace(0, num_angles-1, num_angles)
        # video_str = get_time_str()
        video_str = f'seed_{seed_val}_angles_{dirction_val}'
        out = init_video_writer(video_str)
        mp = MotionPlanner(max_random, min_dist)
        previous_robot_coords = []
        for i in range(num_step):
            print(i)
            board = get_new_board()
            draw = ImageDraw.Draw(board)
            draw_obstacles(draw, obstacle_list)
            image = np.asarray(board)
            dist_map, end_points = measure_dists(image, robot_coord, angle_vectors)
            idx = get_idx_around_angle(robot_angle)
            visible_dists = dist_map[idx]

            available_angle_vectors = angle_vectors[idx]
            available_angle_list = angle_list[idx]
            draw = draw_dists(draw, robot_coord, end_points, idx)
            step_choice = mp.majority_vote(visible_dists, 'random')
            if step_choice != -1:
                robot_angle = available_angle_list[step_choice]
                step_vector = available_angle_vectors[step_choice]
            else:
                step_choice = mp.majority_vote(dist_map, 'random', True)
                robot_angle = angle_list[step_choice]
                step_vector = angle_vectors[step_choice]
            d_col, d_row = step_vector
            [col_robot, row_robot] = robot_coord
            previous_robot_coords.append(np.copy(robot_coord))
            robot_coord = [col_robot + d_col, row_robot + d_row]
        # draw_path(previous_robot_coords, obstacle_list)
            image = prep_board_for_video(board)
            out.write(image)
        out.release()
