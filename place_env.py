import torch
import numpy as np

import os
import sys
import math

from base_placement.no_overlap_base import BasePlacement_NoOverlap
from base_placement.overlap_base import BasePlacement_Overlap

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)


np.set_printoptions(threshold=np.inf)


def compute_intrinsic_reward(rnd, next_obs, mse, optim):
    next_obs = next_obs.cuda()
    target_next_feature = rnd.target(next_obs)
    predict_next_feature = rnd.predictor(next_obs)

    forward_loss = mse(predict_next_feature, target_next_feature).mean(-1)
    intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
    optim.zero_grad()
    forward_loss.backward()

    return intrinsic_reward.item() / 100


class Placememt(BasePlacement_Overlap, BasePlacement_NoOverlap):
    def __init__(self, benchmark, grid_size=32, overlap=True):
        if overlap:
            super(BasePlacement_Overlap  , self).__init__(benchmark, grid_size)
            self.cal_re = self.cal_re_overlap
        else:
            super(BasePlacement_NoOverlap, self).__init__(benchmark, grid_size)
            self.cal_re = self.cal_re_disjoint
        self.overlap = overlap

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
                pl = []
                for res in self.results:
                    x = res[0] / self.n * self.chip_size[0]
                    y = res[1] / self.n * self.chip_size[1]
                    pl.append([int(x), int(y)])
                self.f.write(str(self.obs))
                self.f.write(str(pl))
                self.f.write('\n')
                self.f.write(str(reward))
                self.f.write('\n')
            self.results = []
        else:
            done = False
            reward = compute_intrinsic_reward(rnd, obs / 255.0, mse, optim)
        return obs, done, torch.FloatTensor([[reward]])

#################################################### No Overlap ########################################################

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

##################################################### Overlap ##########################################################

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
