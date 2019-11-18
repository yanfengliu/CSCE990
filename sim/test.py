import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw


# img_size = 600
# board = Image.new(mode="I", size=(img_size, img_size), color=0)
# draw = ImageDraw.Draw(board)
# draw.line((0, 0, 200, 100), fill = 1)
# image = np.asarray(board)
# plt.figure()
# plt.imshow(image)
# plt.show()

min_dist = 5

def majority_vote(dists):
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
    choice = np.argmax(votes)
    target_dist_group = dist_groups[choice]
    target_choice_group = choice_groups[choice]
    direction = target_choice_group[np.argmax(target_dist_group)]
    return direction

print(majority_vote(np.array([10, 2, 8, 7, 6, 3, 4, 9, 10, 11, 12])))