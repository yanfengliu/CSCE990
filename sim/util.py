import numpy as np
import os


def get_random_rect(img_size):
    width = np.random.randint(low = 0.1 * img_size, high = 0.3 * img_size)
    height = np.random.randint(low = 0.1 * img_size, high = 0.3 * img_size)
    corners = [[0, 0], [width, 0], [width, height], [0, height], [0, 0]]
    offset = np.random.randint(low = 0, high = 0.7 * img_size, size=(2,))
    corners += offset
    return corners


def to_tuple(corners):
    return [tuple(i) for i in corners]


def get_dist_at_angle(image, img_size, robot_coord, angle_vector):
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


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)