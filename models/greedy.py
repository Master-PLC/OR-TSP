#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :greedy.py
@Description  :
@Date         :2021/10/13 16:45:22
@Author       :Arctic Little Pig
@Version      :1.0
'''

import time
import numpy as np
from typing import List, Tuple


class Greedy(object):
    def __init__(self, location: np.ndarray, dist_matrix: np.ndarray, num_test: int) -> None:
        super().__init__()
        self.location = location
        self.dist_matrix = dist_matrix
        self.N = dist_matrix.shape[0]
        self.num_test = num_test

        self.pathLen = 0
        self.runtime = 0

    def search(self) -> List:
        v_dict = dict()

        for i in range(self.num_test):
            print(f"Number {i} test.")
            
            i = 1
            j = 0
            path_length = 0
            v = []  # 已经遍历过的社区
            v.append(0)

            start_time = time.time()
            while True:
                k = 1
                dist_temp = 10000000  # 当前最小距离
                while True:
                    l = 0
                    flag = 0  # 访问标记
                    if k in v:
                        flag = 1
                    if (flag == 0) and (self.dist_matrix[k, v[i - 1]] < dist_temp):
                        j = k
                        dist_temp = self.dist_matrix[k, v[i - 1]]
                    k += 1
                    if k >= self.N:
                        break
                v.append(j)
                i += 1
                path_length += dist_temp
                if i >= self.N:
                    break
            path_length += self.dist_matrix[0, j]
            v = v+[v[0]]

            end_time = time.time()
            interval = end_time - start_time
            print(f"Runing time: {interval:.5f}.")
            self.runtime += interval
            v_dict[path_length] = v

        sorted_dict = sorted(v_dict.items(), key=lambda item: item[0])
        self.pathLen = sorted_dict[0][0]
        v = sorted_dict[0][1]

        return v
    
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
