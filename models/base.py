#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :base.py
@Description  :
@Date         :2021/10/14 18:50:22
@Author       :Arctic Little Pig
@Version      :1.0
'''

import time
import numpy as np
from typing import List, Tuple


class BaseModel(object):
    def __init__(self, location: np.ndarray, dist_matrix: np.ndarray, num_test: int) -> None:
        super().__init__()
        self.location = location
        self.N = dist_matrix.shape[0]
        self.dist_matrix = dist_matrix
        self.num_test = num_test

        self.pathLen = 0
        self.runtime = 0

    def get_coordinates(self, path: List) -> Tuple:
        X = []
        Y = []

        for v in path:
            X.append(self.location[v, 0])
            Y.append(self.location[v, 1])

        return X, Y

    def get_runtime(self) -> float:
        return self.runtime / self.num_test

    def get_path_length(self) -> float:
        return self.pathLen
