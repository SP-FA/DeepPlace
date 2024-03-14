import torch
from gym.spaces import Discrete
from gym.utils import seeding

from read_benchmarks import generate_db_params


class BasePlacement:
    def __init__(self, benchmark, grid_size):
        self.n = grid_size
        self.benchmark = benchmark
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

        self.steps = len(self.node_info)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.obs = torch.zeros((1, 1, self.n, self.n))
        return self.obs

    def to(self, device):
        self.obs = self.obs.to(device)

    # def transform(self, x):
    #     up = nn.Upsample(size=84, mode='bilinear', align_corners=False)
    #     return up(x)*255
