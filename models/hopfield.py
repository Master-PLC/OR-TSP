#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :hopfield.py
@Description  :
@Date         :2021/10/14 14:56:53
@Author       :Arctic Little Pig
@Version      :1.0
'''

import time
import numpy as np
from typing import List, Tuple

from .base import BaseModel


class Hopfield(BaseModel):
    def __init__(self, location: np.ndarray, dist_matrix: np.ndarray, num_test: int) -> None:
        super(Hopfield, self).__init__(location, dist_matrix, num_test)
        # for i in range(self.N):
        #     dist_matrix[i, i] = np.sqrt(np.sum(location[i, :]**2))

        self.A = 2000
        self.D = 25
        self.u0 = 0.02
        self.step = 0.001
        self.iterations = 4000

    def init_path(self):
        path = list(range(self.N))

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

    def delta_u(self, V):
        t1 = np.tile(np.sum(V, axis=1, keepdims=True)-1, [1, self.N])
        t2 = np.tile(np.sum(V, axis=0, keepdims=True)-1, [self.N, 1])
        PermitV = V[:, 1:self.N]
        PermitV = np.hstack([PermitV, V[:, 0].reshape(self.N, -1)])
        t3 = np.dot(self.dist_matrix, PermitV)
        du = -self.A*(t1+t2)-self.D*t3

        return du

    def energy(self, V):
        t1 = np.sum((np.sum(V, axis=1)-1)**2)
        t2 = np.sum((np.sum(V, axis=0)-1)**2)
        PermitV = V[:, 1:self.N]
        PermitV = np.hstack([PermitV, V[:, 0].reshape(self.N, -1)])
        # temp = np.dot(self.dist_matrix, PermitV)
        temp = self.dist_matrix*PermitV
        t3 = np.sum(V*temp)
        E = 0.5*(self.A*(t1+t2)+self.D*t3)

        return E

    def route_check(self, V):
        route = []
        for i in range(self.N):
            mm = np.max(V[:, i])
            for j in range(self.N):
                if V[j, i] == mm:
                    route += [j]
                    break

        return route

    def search(self) -> List:
        v_dict = dict()

        energy_list = []

        for i in range(self.num_test):
            print(f"Number {i} test.")

            best_distance = np.inf
            path_best = []

            start_time = time.time()

            u = 2*np.random.rand(self.N, self.N)-1
            U = 0.5*self.u0*np.log(self.N-1)+u
            V = (1+np.tanh(U/self.u0))/2

            for iteration in range(self.iterations):
                dU = self.delta_u(V)
                U = U+dU*self.step
                V = (1+np.tanh(U/self.u0))/2
                E = self.energy(V)
                energy_list.append(E)

                route = self.route_check(V)

                # print(len(np.unique(route)))
                if len(np.unique(route)) == self.N:
                    distance = self.cal_path_length(route)
                    if distance < best_distance:
                        best_distance = distance
                        path_best = route

            path_length = self.cal_path_length(path_best)

            zero_index = path_best.index(0)
            v = path_best[zero_index:] + path_best[:zero_index] + [0]

            end_time = time.time()
            interval = end_time - start_time
            print(f"Runing time: {interval:.5f}s.")
            self.runtime += interval
            v_dict[path_length] = v

        sorted_dict = sorted(v_dict.items(), key=lambda item: item[0])
        self.pathLen = sorted_dict[0][0]
        v = sorted_dict[0][1]

        return v