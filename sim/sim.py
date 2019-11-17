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


    while( 
        col_angle < img_size and col_angle >= 0 and 
        row_angle < img_size and row_angle >= 0 and 
        image[row_angle, col_angle] != 1):

        col_old, row_old = col_angle, row_angle
        d_col, d_row = angle_vector
        col_angle += d_col
        row_angle += d_row

    dist_v = row_old - row_robot
    dist_h = col_old - col_robot
    dist = np.sqrt(dist_h ** 2 + dist_v ** 2)
    return dist, col_old, row_old

# constants
img_size = 1000
robot_size = 20
num_step = 1
max_obstacle = 8
view_angle_range = 90
num_obstacle = np.random.randint(1, max_obstacle)

# create map with obstacle
board = Image.new(mode="I", size=(img_size, img_size), color=0)
draw = ImageDraw.Draw(board)
obstacle_list = []
for i in range(num_obstacle):
    corners = get_random_rect(img_size)
    obstacle_list.append(corners)
    shape_tuple = to_tuple(corners)
    draw.polygon(xy=shape_tuple, fill=1, outline=1)
image = np.asarray(board)

# get empty space
empty_coords = []
for row in range(img_size):
    for col in range(img_size):
        if not image[row, col]:
            empty_coords.append([col, row])
# init robot in empty space with random coord and angle
robot_coord_choice = np.random.choice(len(empty_coords))
robot_coord = np.array(empty_coords[robot_coord_choice])
corners = [
    robot_coord, 
    robot_coord + np.array([robot_size, 0]), 
    robot_coord + np.array([robot_size, robot_size]), 
    robot_coord + np.array([0, robot_size])
    ]
shape_tuple = to_tuple(corners)
draw.polygon(xy=shape_tuple, fill=2, outline=2)
image = np.asarray(board)
image = np.copy(image)
robot_angle = np.random.randint(360)

for i in range(num_step):
    print('here')
    # get dist map
    dist_map = []
    end_points = []
    
    # angles arranged in the order of :
    #        0 15 14 13 12 
    #        1          11
    #        2          10
    #        3           9
    #        4  5  6  7  8
    angle_vectors = [
        [-2, -2], [-2, -1], [-2,  0], [-2,  1], 
        [-2,  2], [-1,  2], [ 0,  2], [ 1,  2], 
        [ 2,  2], [ 2,  1], [ 2,  0], [ 2, -1], 
        [ 2, -2], [ 1, -2], [ 0, -2], [-1, -2]
    ]

    for angle_vector in angle_vectors:
        dist, col_end, row_end = get_dist(image, robot_coord, angle_vector)
        dist_map.append(dist)
        end_points.append([col_end, row_end])
    
    print('dist map:')
    for dist in dist_map:
        print(dist)
    print('end_points:')
    for end_point in end_points:
        print(end_point)
    [col_robot, row_robot] = robot_coord
    for end_point in end_points:
        col_end, row_end = end_point
        draw.line((col_robot, row_robot, col_end, row_end), fill=3, width = 5)
        image = np.asarray(board)
        image = np.copy(image)

    plt.figure()
    plt.imshow(image)
    plt.title('map')

    plt.figure()
    plt.plot(dist_map)
    plt.title('dist map')
    plt.show()
