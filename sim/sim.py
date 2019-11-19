from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw


def get_random_rect(img_size):
    width = np.random.randint(low = 0.1 * img_size, high = 0.3 * img_size)
    height = np.random.randint(low = 0.1 * img_size, high = 0.3 * img_size)
    corners = [[0, 0], [width, 0], [width, height], [0, height], [0, 0]]
    offset = np.random.randint(low = 0, high = 0.7 * img_size, size=(2,))
    corners += offset
    return corners


def to_tuple(corners):
    return [tuple(i) for i in corners]


def get_dist(image, robot_coord, angle_vector):
    [col_robot, row_robot] = robot_coord
    [col_old, row_old] = robot_coord
    [col_angle, row_angle] = robot_coord
    count = 0
    while( 
        col_angle < img_size and col_angle >= 0 and 
        row_angle < img_size and row_angle >= 0 and 
        np.all(image[row_angle, col_angle, :] != np.array([255, 255, 255]))):

        count += 1
        col_old, row_old = col_angle, row_angle
        d_col, d_row = angle_vector
        col_angle += d_col
        row_angle += d_row
    dist_v = row_old - row_robot
    dist_h = col_old - col_robot
    dist = np.sqrt(dist_h ** 2 + dist_v ** 2)

    return dist, col_old, row_old


class MotionPlanner():
    def __init__(self, max_random):
        self.max_random = max_random
        self.counter = 0

    def majority_vote(self, dists, mode, override_random_counter=False):
        num_dist = len(dists)
        choices = np.linspace(0, num_dist-1, num_dist)
        dist_groups = []
        dist_group = []
        choice_groups = []
        choice_group = []
        for i in range(num_dist):
            dist = dists[i]
            if dist > min_dist:
                dist_group.append(dist)
                choice_group.append(i)
            elif len(dist_group) > 0:
                dist_groups.append(dist_group)
                choice_groups.append(choice_group)
                dist_group = []
                choice_group = []
        if len(dist_group) > 0:
            dist_groups.append(dist_group)
            choice_groups.append(choice_group)
        votes = []
        for dist_group in dist_groups:
            votes.append(len(dist_group))
        votes = np.array(votes)
        if len(votes) > 0:
            choice = np.argmax(votes)
            target_dist_group = dist_groups[choice]
            target_choice_group = choice_groups[choice]
            if mode == 'max':
                idx = np.argmax(target_dist_group)
            elif mode == 'random':
                if not override_random_counter:
                    if self.counter <= self.max_random:
                        self.counter = 0
                        idx = np.argmax(target_dist_group)
                    else:
                        self.counter += 1
                        idx = np.random.randint(low = 0, high=len(target_dist_group))
                else:
                    idx = np.random.randint(low = 0, high=len(target_dist_group))
            direction = target_choice_group[idx]
        else:
            direction = -1
        return direction

# constants
img_size = 500
robot_size = 5
num_step = 2000
num_angles = 16
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

mp = MotionPlanner(max_random)

# NOTE: nparray is (row, col), ImageDraw is (col, row)

# create map with obstacle
board = Image.new(mode="RGB", size=(img_size, img_size), color=BACKGROUND_COLOR)
draw = ImageDraw.Draw(board)
obstacle_list = []
for i in range(num_obstacle):
    corners = get_random_rect(img_size)
    obstacle_list.append(corners)
    shape_tuple = to_tuple(corners)
    draw.polygon(xy=shape_tuple, fill=OBSTACLE_COLOR, outline=OBSTACLE_COLOR)
image = np.asarray(board)

# get empty space
empty_coords = []
for row in range(img_size):
    for col in range(img_size):
        if not np.all(image[row, col, :] == np.array(OBSTACLE_COLOR)):
            empty_coords.append([col, row])
# init robot in empty space with random coord and angle
robot_coord_choice = np.random.choice(len(empty_coords))
robot_coord = np.array(empty_coords[robot_coord_choice])
corners = [
    robot_coord - [robot_size, robot_size], 
    robot_coord + [robot_size, -robot_size], 
    robot_coord + [robot_size, robot_size], 
    robot_coord + [-robot_size, robot_size]
    ]
shape_tuple = to_tuple(corners)
draw.polygon(xy=shape_tuple, fill=ROBOT_COLOR, outline=ROBOT_COLOR)

robot_angle = np.random.randint(num_angles)
angle_list = np.linspace(0, num_angles-1, num_angles)

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
out = cv2.VideoWriter(f'simulation_v1_{dt_string}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (img_size, img_size))

for i in range(num_step):
    print(i)
    board = Image.new(mode="RGB", size=(img_size, img_size), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(board)
    for i in range(num_obstacle):
        corners = obstacle_list[i]
        shape_tuple = to_tuple(corners)
        draw.polygon(xy=shape_tuple, fill=OBSTACLE_COLOR, outline=OBSTACLE_COLOR)
    image = np.asarray(board)

    dist_map = []
    end_points = []
    
    # angles arranged in the order of :
    #        0 15 14 13 12 
    #        1          11
    #        2          10
    #        3           9
    #        4  5  6  7  8

    angle_vectors = np.array([
        [-2, -2], [-2, -1], [-2,  0], [-2,  1], 
        [-2,  2], [-1,  2], [ 0,  2], [ 1,  2], 
        [ 2,  2], [ 2,  1], [ 2,  0], [ 2, -1], 
        [ 2, -2], [ 1, -2], [ 0, -2], [-1, -2]
    ])

    for angle_vector in angle_vectors:
        dist, col_end, row_end = get_dist(image, robot_coord, angle_vector)
        dist_map.append(dist)
        end_points.append([col_end, row_end])

    dist_map = np.array(dist_map)
    dist_idx = robot_angle + np.array([-2, -1, 0, 1, 2])
    dist_idx = dist_idx % num_angles
    dist_idx = dist_idx.astype(np.int)
    visible_dists = dist_map[dist_idx]
    available_angle_vectors = angle_vectors[dist_idx]
    available_angle_list = angle_list[dist_idx]
    [col_robot, row_robot] = robot_coord
    for j in dist_idx:
        col_end, row_end = end_points[j]
        draw.line((col_robot, row_robot, col_end, row_end), fill=DISTANCE_COLOR, width = 3)
        if dist_map[j] <= min_dist:
            draw.line((col_robot, row_robot, col_end, row_end), fill=WARNING_COLOR, width = 3)
    step_choice = mp.majority_vote(visible_dists, 'random')
    if step_choice != -1:
        robot_angle = available_angle_list[step_choice]
        step_vector = available_angle_vectors[step_choice]
    else:
        step_choice = mp.majority_vote(dist_map, 'random', True)
        robot_angle = angle_list[step_choice]
        step_vector = angle_vectors[step_choice]
    d_col, d_row = step_vector
    robot_coord = [col_robot + d_col, row_robot + d_row]
    image = np.asarray(board)
    image = np.copy(image)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out.write(image)
out.release()
