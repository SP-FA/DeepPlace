import torch

from base_placement.base import BasePlacement


class BasePlacement_Overlap(BasePlacement):
    def __init__(self, benchmark, grid_size):
        super().__init__(benchmark, grid_size)

    def is_valid(self, x, y):
        if -1 < x < self.n and -1 < y < self.n and self.obs[0, 0, x, y] < 1.0:
            return True
        return False

    def search(self, x, y, _, __, depth):
        x = x.cpu()
        y = y.cpu()
        if self.obs[0, 0, x, y] < 1.0:
            return x, y
        if depth > 7:
            return -1, -1
        elif self.is_valid(x - 1, y): return x - 1, y
        elif self.is_valid(x + 1, y): return x + 1, y
        elif self.is_valid(x, y - 1): return x, y - 1
        elif self.is_valid(x, y + 1): return x, y + 1
        else:
            return self.search(x - 1, y - 1, _, __, depth + 1)

    def find(self, _, __):
        center = [self.n // 2, self.n // 2]
        ob = self.obs[0, 0]
        for i in range(self.n):
            for j in range(i):
                if self.is_valid(center[0] - j, center[1] - (i - j)):
                    return center[0] - j, center[1] - (i - j)
                if self.is_valid(center[0] - j, center[1] + (i - j)):
                    return center[0] - j, center[1] + (i - j)
                if self.is_valid(center[0] + j, center[1] - (i - j)):
                    return center[0] + j, center[1] - (i - j)
                if self.is_valid(center[0] + j, center[1] + (i - j)):
                    return center[0] + j, center[1] + (i - j)
