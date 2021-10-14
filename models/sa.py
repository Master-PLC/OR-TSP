#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :sa.py
@Description  :
@Date         :2021/10/14 11:20:48
@Author       :Arctic Little Pig
@Version      :1.0
'''

import time
import math
import random
import numpy as np
from typing import List, Tuple

from .base import BaseModel


class SA(BaseModel):
    def __init__(self, location: np.ndarray, dist_matrix: np.ndarray, num_test: int) -> None:
        super(SA, self).__init__(location, dist_matrix, num_test)

        self.iteration1 = 1000  # 外循环迭代次数
        self.T0 = 100000  # 初始温度，取大些
        self.Tf = 1  # 截止温度，可以不用
        self.alpha = 0.95  # 温度更新因子
        self.iteration2 = 10  # 内循环迭代次数

    def init_path(self):
        path = random.sample(range(self.N), self.N)

        return path

    def cal_path_length(self, code_vector):
        path_length = 0
        distance = 0
        for i in range(self.N):
            if i < self.N-1:
                # 计算距离
                distance = self.dist_matrix[code_vector[i], code_vector[i+1]]
            else:
                distance = self.dist_matrix[code_vector[i], code_vector[0]]
            path_length += distance

        return path_length

    def search(self) -> List:
        v_dict = dict()

        for i in range(self.num_test):
            # print(f"Number {i} test.")

            start_time = time.time()
            init_path = self.init_path()
            fbest = self.cal_path_length(init_path)

            path_best = init_path.copy()
            f_now = fbest
            path_now = path_best.copy()

            for i in range(self.iteration1):
                for k in range(self.iteration2):
                    # 生成新解
                    path1 = [0] * self.N
                    n = [random.randint(0, self.N-1),
                         random.randint(0, self.N-1)]
                    n.sort()
                    n1, n2 = n
                    # n1为0单独写
                    if n1 > 0:
                        path1[0:n1] = path_now[0:n1]
                        path1[n1:n2+1] = path_now[n2:n1-1:-1]
                        path1[n2+1:self.N] = path_now[n2+1:self.N]
                    else:
                        path1[0:n1] = path_now[0:n1]
                        path1[n1:n2+1] = path_now[n2::-1]
                        path1[n2+1:self.N] = path_now[n2+1:self.N]

                    s = self.cal_path_length(path1)
                    # 判断是否更新解
                    if s <= f_now:
                        f_now = s
                        path_now = path1.copy()
                    else:
                        deltaf = s - f_now
                        if random.random() < math.exp(-deltaf/self.T0):
                            f_now = s
                            path_now = path1.copy()

                    if s < fbest:
                        fbest = s
                        path_best = path1.copy()
                self.T0 *= self.alpha

            zero_index = path_best.index(0)
            v = path_best[zero_index:] + path_best[:zero_index] + [0]
            path_length = fbest

            end_time = time.time()
            interval = end_time - start_time
            # print(f"Runing time: {interval:.5f}s.")
            self.runtime += interval
            v_dict[path_length] = v

        sorted_dict = sorted(v_dict.items(), key=lambda item: item[0])
        self.pathLen = sorted_dict[0][0]
        v = sorted_dict[0][1]

        return v
