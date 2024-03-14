import torch

from base_placement.base import BasePlacement


class BasePlacement_NoOverlap(BasePlacement):
    def __init__(self, benchmark, grid_size):
        super().__init__(benchmark, grid_size)
        self.mask = torch.zeros((self.n, self.n))

    def reset(self):
        self.obs = torch.zeros((1, 1, self.n, self.n))
        self.mask = torch.zeros((self.n, self.n))
        return self.obs

    def is_valid(self, x, y, shift_w, shift_h):
        if -1 < x < self.n and -1 < y < self.n and -1 < x + shift_w < self.n and -1 < y + shift_h < self.n:
            if torch.sum(self.mask[x:x + shift_w, y:y + shift_h]) == 0:
                return True
        return False

    def update_mask(self, x, y, shift_w, shift_h):
        if -1 < x < self.n and -1 < y < self.n and -1 < x + shift_w < self.n and -1 < y + shift_h < self.n:
            self.mask[x:x + shift_w, y:y + shift_h] = 1

    def search(self, x, y, shift_w, shift_h, depth):
        if self.is_valid(x, y, shift_w, shift_h): return x, y
        if depth > 7:
            return -1, -1
        elif self.is_valid(x - 1, y, shift_w, shift_h): return x - 1, y
        elif self.is_valid(x + 1, y, shift_w, shift_h): return x + 1, y
        elif self.is_valid(x, y - 1, shift_w, shift_h): return x, y - 1
        elif self.is_valid(x, y + 1, shift_w, shift_h): return x, y + 1
        else:
            return self.search(x - 1, y - 1, shift_w, shift_h, depth + 1)

    def find(self, shift_w, shift_h):
        midx = midy = self.n // 2
        for r in range(self.n):
            for x in range(r):
                y = r - x
                if self.is_valid(midx - x, midy - y, shift_w, shift_h): return midx - x, midy - y
                if self.is_valid(midx - x, midy + y, shift_w, shift_h): return midx - x, midy + y
                if self.is_valid(midx + x, midy - y, shift_w, shift_h): return midx + x, midy - y
                if self.is_valid(midx + x, midy + y, shift_w, shift_h): return midx + x, midy + y
