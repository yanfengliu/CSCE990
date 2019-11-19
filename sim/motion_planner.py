import numpy as np


class MotionPlanner():
    def __init__(self, max_random, min_dist):
        self.max_random = max_random
        self.min_dist = min_dist
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
            if dist > self.min_dist:
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

    
    def majority_vote_weighted_sum(self, dists, angle_vectors):
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
            normalized_dist = target_dist_group / np.sum(target_dist_group)
            target_choice_group = np.array(target_choice_group)
            new_robot_angle = int(np.sum(target_choice_group * normalized_dist))
            normalized_dist = np.expand_dims(normalized_dist, axis=-1)
            target_angle_vectors = angle_vectors[target_choice_group]
            direction = normalized_dist * target_angle_vectors
            direction = np.sum(direction, 0)
            return direction, new_robot_angle
        else:
            return None