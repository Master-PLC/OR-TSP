#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :branchbound.py
@Description  :
@Date         :2021/10/13 20:26:36
@Author       :Arctic Little Pig
@Version      :1.0
'''

import time
import numpy as np
from queue import Queue
from typing import List, Tuple


class BranchBound(object):
    def __init__(self, location: np.ndarray, dist_matrix: np.ndarray, num_test: int) -> None:
        super().__init__()
        self.location = location
        self.N = dist_matrix.shape[0]
        for i in range(self.N):
            dist_matrix[i, i] = np.inf
        self.dist_matrix = dist_matrix
        self.num_test = num_test

        self.pathLen = 0
        self.runtime = 0

    def init_bound_queue(self):
        self.low = 0
        self.up = 0

        self.pq = Queue()
        self.dfs_visited = np.zeros(self.N)
        self.dfs_visited[0] = 1

    def dfs(self, u, k, l):
        if k == self.N-1:
            return l + self.dist_matrix[u, 0]

        minlen = np.inf
        p = 0
        for i in range(self.N):
            if self.dfs_visited[i] == 0 and minlen > self.dist_matrix[u, i]:
                minlen = self.dist_matrix[u, i]
                p = i
        self.dfs_visited[p] = 1

        return self.dfs(p, k+1, l+minlen)

    def get_up(self):
        self.up = self.dfs(0, 0, 0)

    def get_low(self):
        for i in range(self.N):
            temp = self.dist_matrix[i].copy()
            temp.sort()
            self.low = self.low + temp[0] + temp[1]
        self.low = self.low / 2

    def get_lb(self, p):
        ret = p.sumv*2
        min1 = np.inf  # 起点和终点连出来的边
        min2 = np.inf
        # 从起点到最近未遍历城市的距离
        for i in range(self.N):
            if p.visited[i] == 0 and min1 > self.dist_matrix[i, p.start]:
                min1 = self.dist_matrix[i, p.start]
        ret = ret + min1

        # 从终点到最近未遍历城市的距离
        for j in range(self.N):
            if p.visited[j] == 0 and min2 > self.dist_matrix[p.end, j]:
                min2 = self.dist_matrix[p.end, j]
        # 进入并离开每个未遍历城市的最小成本
        for i in range(self.N):
            if p.visited[i] == 0:
                min1 = min2 = np.inf
                for j in range(self.N):
                    min1 = self.dist_matrix[i,
                                            j] if min1 > self.dist_matrix[i, j] else min1
                for m in range(self.N):
                    min2 = self.dist_matrix[i,
                                            m] if min2 > self.dist_matrix[m, i] else min2
                ret = ret + min1 + min2

        return (ret+1) / 2

    def search(self) -> List:
        v_dict = dict()

        for i in range(self.num_test):
            print(f"Number {i} test.")

            self.init_bound_queue()

            start_time = time.time()

            self.get_up()
            self.get_low()  # 获得下界

            node = Node(self.N)
            node.start = 0  # 起始点从1开始
            node.end = 0  # 结束点到1结束(当前路径的结束点)
            node.listc.append(0)
            node.visited[0] = 1
            node.lb = self.low  # 初始目标值等于下界
            path_length = np.inf  # path_length是问题的最终解
            self.pq.put(node)  # 将起点加入队列

            while self.pq.qsize() != 0:  # 如果已经走过了n-1个点
                tmp = self.pq.get()
                if tmp.k == self.N-1:
                    p = 0  # 最后一个没有走的点
                    for i in range(self.N):
                        if tmp.visited[i] == 0:
                            p = i
                            break
                    ans = tmp.sumv + \
                        self.dist_matrix[tmp.start, p] + \
                        self.dist_matrix[p, tmp.end]  # 总的路径消耗
                    # 如果当前的路径和比所有的目标函数值都小则跳出
                    # 否则继续求其他可能的路径和，并更新上界
                    if ans <= tmp.lb:
                        path_length = min(ans, path_length)
                        break
                    else:
                        self.up = min(ans, self.up)  # 上界更新为更接近目标的ans值
                        path_length = min(path_length, ans)
                        continue
                # 当前点可以向下扩展的点入优先级队列

                for i in range(self.N):
                    if tmp.visited[i] == 0:
                        next_node = Node(self.N)
                        next_node.start = tmp.start    # 沿着tmp走到next，起点不变
                        next_node.sumv = tmp.sumv + \
                            self.dist_matrix[tmp.end, i]
                        next_node.end = i  # 更新最后一个点
                        next_node.k = tmp.k+1
                        next_node.listc = tmp.listc.copy()
                        next_node.listc.append(i)
                        # print(tmp.k)
                        # tmp经过的点也是next经过的点
                        next_node.visited = tmp.visited.copy()
                        next_node.visited[i] = 1
                        next_node.lb = self.get_lb(next_node)  # 求目标函数
                        if next_node.lb >= self.up:
                            continue
                        self.pq.put(next_node)
            tmp.listc.append(0)  # 加入最后一个点

            end_time = time.time()
            interval = end_time - start_time
            print(f"Runing time: {interval:.5f}s.")
            self.runtime += interval
            v_dict[path_length] = tmp.listc.copy()

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


class Node(object):
    def __init__(self, N):
        self.visited = np.zeros(N)
        self.start = 1
        self.end = 1
        self.k = 1
        self.sumv = 0
        self.lb = 0
        self.listc = []
