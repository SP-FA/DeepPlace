from gym.spaces import Discrete
import torch
import torch.nn as nn
import numpy as np
from gym.utils import seeding
import os
import sys
import logging
from read_benchmarks import generate_db_params

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import math

np.set_printoptions(threshold=np.inf)


def compute_intrinsic_reward(rnd, next_obs, forward_mse, optimizer):
    next_obs = next_obs.cuda()
    target_next_feature = rnd.target(next_obs)
    predict_next_feature = rnd.predictor(next_obs)

    forward_loss = forward_mse(predict_next_feature, target_next_feature).mean(-1)
    intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
    optimizer.zero_grad()
    forward_loss.backward()

    return intrinsic_reward.item() / 100


class Placememt():
    def __init__(self, benchmark, grid_size=32, overlap=True):
        self.n = grid_size
        self.action_space = Discrete(self.n * self.n)
        self.obs_space = (1, self.n, self.n)
        self.obs = torch.zeros((1, 1, self.n, self.n))
        self.results = []
        self.best = -1e9

        (self.node_info,
         self.net_info,
         self.node_id_to_name,
         self.net,
         self.netlist_graph,
         self.chip_size) = generate_db_params(benchmark)

        self.f = open("./result/result.txt", 'w')

        # f = open("./data/n_edges_710.dat", "r")
        # for line in f:
        #     self.net = eval(line)

        self.steps = len(self.node_info)
        self.mask = torch.zeros((self.chip_size[0], self.chip_size[1]))
        self.overlap = overlap

        if overlap:
            self.cal_re = self.cal_re_overlap
            self.search = self.search_overlap
            self.find = self.find_overlap
        else:
            self.cal_re = self.cal_re_disjoint
            self.search = self.search_disjoint
            self.find = self.find_disjoint

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.obs = torch.zeros((1, 1, self.n, self.n))
        self.mask = torch.zeros((self.chip_size[0], self.chip_size[1]))
        return self.obs

    def to(self, device):
        self.obs = self.obs.to(device)

    # def transform(self, x):
    #     up = nn.Upsample(size=84, mode='bilinear', align_corners=False)
    #     return up(x)*255

    def step(self, action, rnd, mse, optim):
        cur_node_info = self.node_info[self.node_id_to_name[len(self.results)]]
        macro_w = cur_node_info["x"]
        macro_h = cur_node_info["y"]

        shift_w = math.ceil(macro_w / self.chip_size[0] * self.n)
        shift_h = math.ceil(macro_h / self.chip_size[1] * self.n)
        # print(f"{len(self.results)=}")
        # print(f"{shift_w=}")
        # print(f"{shift_h=}")

        x = action // self.n
        y = action % self.n
        x, y = self.search(x, y, shift_w, shift_h, 0)
        if x == -1 or y == -1:
            x, y = self.find(shift_w, shift_h)

        if self.overlap:
            self.obs[0, 0, x, y] = 1
        else:
            self.obs[0, 0, x:x + shift_w, y:y + shift_h] = 1
            self.update_mask(x, y, shift_w, shift_h)
        self.results.append([int(x), int(y)])
        # obs = self.transform(self.obs)
        obs = self.obs

        if len(self.results) == self.steps:
            done = True
            reward = self.cal_re()
            if reward > self.best:
                self.best = reward
                self.f.write(str(self.obs))
                self.f.write(str(self.results))
                self.f.write('\n')
                self.f.write(str(reward))
                self.f.write('\n')
            self.results = []
        else:
            done = False
            reward = compute_intrinsic_reward(rnd, obs / 255.0, mse, optim)
        return obs, done, torch.FloatTensor([[reward]])

    def cal_re_disjoint(self):
        wl = 0
        for net_name in self.net_info:
            nodes_in_net = self.net_info[net_name]["nodes"]
            if len(nodes_in_net) <= 1:
                continue

            # collect pins in one macro-net
            pin_position_list = []  # all pins in one macro-net
            for node_name in nodes_in_net:
                raw_center = [0, 0]
                node_id = self.node_info[node_name]["id"]
                center = self.results[node_id]
                raw_center[0] = center[0] * self.chip_size[0] / self.n
                raw_center[1] = center[1] * self.chip_size[1] / self.n
                offset = [self.net_info[net_name]["nodes"][node_name]["x_offset"],
                          self.net_info[net_name]["nodes"][node_name]["y_offset"]]
                pin_position = (raw_center[0] + offset[0], raw_center[1] + offset[1])
                pin_position_list.append(pin_position)

            left = self.chip_size[0]
            right = 0
            up = self.chip_size[1]
            down = 0
            for (x, y) in pin_position_list:
                right = max(right, x)
                left = min(left, x)
                up = min(up, y)
                down = max(down, y)
            wn = int(right - left + 1)
            hn = int(down - up + 1)
            wl += wn + hn
        return -wl / 100000

    def is_valid_disjoint(self, x, y, shift_w, shift_h):
        if -1 < x < self.n and -1 < y < self.n and -1 < x + shift_w < self.n and -1 < y + shift_h < self.n:
            if torch.sum(self.mask[x:x + shift_w, y:y + shift_h]) == 0:
                return True
        return False

    def update_mask(self, x, y, shift_w, shift_h):
        if -1 < x < self.n and -1 < y < self.n and -1 < x + shift_w < self.n and -1 < y + shift_h < self.n:
            self.mask[x:x + shift_w, y:y + shift_h] = 1

    def search_disjoint(self, x, y, shift_w, shift_h, depth):
        if self.is_valid_disjoint(x, y, shift_w, shift_h): return x, y
        if depth > 7:
            return -1, -1
        elif self.is_valid_disjoint(x - 1, y, shift_w, shift_h):
            return x - 1, y
        elif self.is_valid_disjoint(x + 1, y, shift_w, shift_h):
            return x + 1, y
        elif self.is_valid_disjoint(x, y - 1, shift_w, shift_h):
            return x, y - 1
        elif self.is_valid_disjoint(x, y + 1, shift_w, shift_h):
            return x, y + 1
        else:
            return self.search_disjoint(x - 1, y - 1, shift_w, shift_h, depth + 1)

    def find_disjoint(self, shift_w, shift_h):
        midx = midy = self.n // 2
        for r in range(self.n):
            for x in range(r):
                y = r - x
                if self.is_valid_disjoint(midx - x, midy - y, shift_w, shift_h): return midx - x, midy - y
                if self.is_valid_disjoint(midx - x, midy + y, shift_w, shift_h): return midx - x, midy + y
                if self.is_valid_disjoint(midx + x, midy - y, shift_w, shift_h): return midx + x, midy - y
                if self.is_valid_disjoint(midx + x, midy + y, shift_w, shift_h): return midx + x, midy + y

    def is_valid_overlap(self, x, y, _, __):
        if -1 < x < self.n and -1 < y < self.n:
            return True
        return False

    def search_overlap(self, x, y, _, __, depth):
        x = x.cuda()
        y = y.cuda()
        ob = self.obs[0, 0].cuda()
        if ob[x, y] < 1.0:
            return x, y
        if depth > 7:
            return -1, -1
        elif x - 1 >= 0 and ob[x - 1, y] < 1.0:
            return x - 1, y
        elif x + 1 < self.n and ob[x + 1, y] < 1.0:
            return x + 1, y
        elif y - 1 >= 0 and ob[x, y - 1] < 1.0:
            return x, y - 1
        elif y + 1 < self.n and ob[x, y + 1] < 1.0:
            return x, y + 1
        else:
            return self.search_overlap(x - 1, y - 1, _, __, depth + 1)

    def find_overlap(self, _, __):
        center = [self.n // 2, self.n // 2]
        ob = self.obs[0, 0]
        for i in range(self.n):
            for j in range(i):
                if self.is_valid_overlap(center[0] - j, center[1] - (i - j), _, __) and ob[
                    center[0] - j, center[1] - (i - j)] < 1.0:
                    return center[0] - j, center[1] - (i - j)
                if self.is_valid_overlap(center[0] - j, center[1] + (i - j), _, __) and ob[
                    center[0] - j, center[1] + (i - j)] < 1.0:
                    return center[0] - j, center[1] + (i - j)
                if self.is_valid_overlap(center[0] + j, center[1] - (i - j), _, __) and ob[
                    center[0] + j, center[1] - (i - j)] < 1.0:
                    return center[0] + j, center[1] - (i - j)
                if self.is_valid_overlap(center[0] + j, center[1] + (i - j), _, __) and ob[
                    center[0] + j, center[1] + (i - j)] < 1.0:
                    return center[0] + j, center[1] + (i - j)

    def cal_re_overlap(self):
        wl = 0
        con = np.zeros((self.n, self.n))
        for net in self.net:
            left = self.n
            right = 0
            up = self.n
            down = 0
            for i in net:
                left = min(left, self.results[i][1])
                right = max(right, self.results[i][1])
                up = min(up, self.results[i][0])
                down = max(down, self.results[i][0])
            wn = int(right - left + 1)
            hn = int(down - up + 1)
            dn = (wn + hn) / (wn * hn)
            con[up:down + 1, left:right + 1] += dn
            wl += wn + hn
        con = list(con.flatten())
        con.sort(reverse=True)
        return (-np.mean(con[:32]) - (wl - 34000) * 0.1) * 0.2


def place_envs(benchmark, grid_size, overlap):
    return Placememt(benchmark, grid_size, overlap)
