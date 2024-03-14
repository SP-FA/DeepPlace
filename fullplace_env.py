import math
import torch
import numpy as np
import os
import sys
import logging

from base_placement.no_overlap_base import BasePlacement_NoOverlap
from base_placement.overlap_base import BasePlacement_Overlap

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import dreamplace.configure as configure
import Params
import PlaceDB
import NonLinearPlace


np.set_printoptions(threshold=np.inf)


def compute_intrinsic_reward(rnd, next_obs, mse, optim):
    target_next_feature = rnd.target(next_obs)
    predict_next_feature = rnd.predictor(next_obs)

    forward_loss = mse(predict_next_feature, target_next_feature).mean(-1)
    intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
    optim.zero_grad()
    forward_loss.backward()

    return intrinsic_reward.item() / 200


class Placememt(BasePlacement_Overlap, BasePlacement_NoOverlap):
    def __init__(self, benchmark, grid_size=32, overlap=True):
        if overlap:
            super(BasePlacement_Overlap  , self).__init__(benchmark, grid_size)
            self.cal_re = self.cal_re_overlap
        else:
            super(BasePlacement_NoOverlap, self).__init__(benchmark, grid_size)
            self.cal_re = self.cal_re_disjoint
        self.overlap = overlap

        logging.root.name = 'DREAMPlace'
        self.params = Params.Params()
        add = "test/ispd2005/" + benchmark + ".json"
        self.params.load(add)
        os.environ["OMP_NUM_THREADS"] = "%d" % (self.params.num_threads)

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
        # x, y = search(self.obs[0, 0], x, y, 0, self.n)
        if x == -1 or y == -1:
            # x, y = find(self.obs[0, 0], self.n)
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
            # reward = new_cal_re(self.results, self.params)
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

    def write(self):
        f = open(f"./benchmarks/ispd2005/{self.benchmark}/{self.benchmark}", "w")
        with open(f"./data/{self.benchmark}", "r") as f2:
            for line in f2:
                line = line.strip()
                l = line.split()
                if line and l[0][0] == 'o':
                    node_name = l[0]
                    node_id = self.node_info[node_name]["id"]
                    pos = self.results[node_id]
                    x = int(pos[0] / self.n * self.chip_size[0])
                    y = int(pos[1] / self.n * self.chip_size[1])
                    l[1] = str(x)
                    l[2] = str(y)
                    line = '\t'.join(l)
                f.write(line)
                f.write('\n')

    def place(self):
        assert (not self.params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"

        np.random.seed(self.params.random_seed)
        placedb = PlaceDB.PlaceDB()
        placedb(self.params)

        placer = NonLinearPlace.NonLinearPlace(self.params, placedb)
        metrics = placer(self.params, placedb)
        result = metrics[-3][0]

        # write placement solution
        path = "%s/%s" % (self.params.result_dir, self.params.design_name())
        if not os.path.exists(path):
            os.system("mkdir -p %s" % (path))
        gp_out_file = os.path.join(
            path,
            "%s.gp.%s" % (self.params.design_name(), self.params.solution_file_suffix()))
        placedb.write(self.params, gp_out_file)
        return result

#################################################### No Overlap ########################################################

    def cal_re_disjoint(self, params):
        self.write()
        r = self.place()
        wl = float(r[0].hpwl.data)
        return -wl / 10000000

##################################################### Overlap ##########################################################

    def cal_re_overlap(self, params):
        self.write()
        r = self.place()
        wl = float(r[0].hpwl.data)
        overf = float(r[0].overflow.data)
        reward = -2 * (wl - 2.4e8) * 1e-6 - overf * 20
        return reward


def fullplace_envs(benchmark, grid_size, overlap):
    return Placememt(benchmark, grid_size, overlap)
