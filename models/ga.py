#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :ga.py
@Description  :
@Date         :2021/10/14 08:22:07
@Author       :Arctic Little Pig
@Version      :1.0
'''

import time
import random
import numpy as np
import pandas as pd
from typing import List, Tuple


__all__ = ("GA_V1", )


class GA_V1(object):
    def __init__(self, location: np.ndarray, dist_matrix: np.ndarray, num_test: int) -> None:
        super().__init__()
        self.location = location
        self.dist_matrix = dist_matrix
        self.N = dist_matrix.shape[0]
        self.num_test = num_test

        self.iterations = 100  # 迭代次数
        self.popsize = 100  # 种群大小
        self.tournament_size = 5  # 锦标赛小组大小
        self.crossover_pr = 0.95  # 交叉概率
        self.mutation_pr = 0.1  # 变异概率

        self.pathLen = 0
        self.runtime = 0

    def init_pop(self):
        return [random.sample(list(range(self.N)), self.N)
                for j in range(self.popsize)]

    def cal_fitness(self, code_vector):
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

    def tournament_select(self, pop, fitness):
        new_pop, new_fits = [], []
        while len(new_pop) < len(pop):
            tournament_list = random.sample(
                range(self.popsize), self.tournament_size)
            tournament_fit = [fitness[i] for i in tournament_list]
            # 转化为df方便索引
            tournament_df = pd.DataFrame([tournament_list, tournament_fit]).transpose(
            ).sort_values(by=1).reset_index(drop=True)
            # print(tournament_df)
            # 选出获胜者
            fit = tournament_df.iloc[0, 1]
            ind = pop[int(tournament_df.iloc[0, 0])]
            new_pop.append(ind)
            new_fits.append(fit)

        return new_pop, new_fits

    def crossover(self, parent1_pop, parent2_pop):
        child_pop = []
        for i in range(self.popsize):
            # 初始化
            child = [0] * self.N
            parent1 = parent1_pop[i]
            parent2 = parent2_pop[i]
            if random.random() >= self.crossover_pr:
                child = parent1.copy()  # 随机生成一个（或者随机保留父代中的一个）
                random.shuffle(child)
            else:
                # parent1
                start_pos = random.randint(0, self.N-1)
                end_pos = random.randint(0, self.N-1)
                if start_pos > end_pos:
                    temp_pos = start_pos
                    start_pos = end_pos
                    end_pos = temp_pos

                child[start_pos:end_pos+1] = parent1[start_pos:end_pos+1].copy()
                # parent2 -> child
                list1 = list(range(end_pos+1, self.N))
                list2 = list(range(start_pos))
                list_index = list1 + list2
                j = -1
                for i in list_index:
                    for j in range(j+1, self.N):
                        if parent2[j] not in child:
                            child[i] = parent2[j]
                            break

            child_pop.append(child)

        return child_pop

    def mutate(self, pop):
        pop_mutate = []
        for i in range(self.popsize):
            ind = pop[i].copy()
            # 随机多次成对变异
            mutate_time = random.randint(1, 5)
            count = 0
            while count < mutate_time:
                if random.random() < self.mutation_pr:
                    mut_pos1 = random.randint(0, self.N-1)
                    mut_pos2 = random.randint(0, self.N-1)
                    if mut_pos1 != mut_pos2:
                        temp = ind[mut_pos1]
                        ind[mut_pos1] = ind[mut_pos2]
                        ind[mut_pos2] = temp
                pop_mutate.append(ind)
                count += 1

        return pop_mutate

    def search(self) -> List:
        v_dict = dict()

        for i in range(self.num_test):
            print(f"Number {i} test.")

            start_time = time.time()

            iteration = 0
            best_fit_list = []
            pop = self.init_pop()

            fitness = [0] * self.popsize
            for i in range(self.popsize):
                fitness[i] = self.cal_fitness(pop[i])
            # print(fitness)
            # 保留当前最优
            best_fit = min(fitness)
            best_pop = pop[fitness.index(best_fit)]
            # print(f'Fitness of the optimal individual in the initial population: {best_fit:.10f}.')
            best_fit_list.append(best_fit)

            while iteration <= self.iterations:
                # 锦标赛赛选择
                pop1, fitness1 = self.tournament_select(pop, fitness)
                pop2, fitness2 = self.tournament_select(pop, fitness)
                # 交叉
                child_pop = self.crossover(pop1, pop2)
                # 变异
                child_pop = self.mutate(child_pop)
                # 计算子代适应度
                child_fits = [0] * self.popsize
                for i in range(self.popsize):
                    child_fits[i] = self.cal_fitness(child_pop[i])
                # 一对一生存者竞争
                # print(fitness, child_fits)
                for i in range(self.popsize):
                    if fitness[i] > child_fits[i]:
                        fitness[i] = child_fits[i]
                        pop[i] = child_pop[i]

                if best_fit > min(fitness):
                    best_fit = min(fitness)
                    best_pop = pop[fitness.index(best_fit)]

                best_fit_list.append(best_fit)

                # print(
                #     f'Fitness of the optimal individual in the {iteration}th iteration: {best_fit:%.1f}.')

                iteration += 1
            zero_index = best_pop.index(0)
            v = best_pop[zero_index:] + best_pop[:zero_index] + [0]
            path_length = best_fit

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


class GA_V2(object):
    def __init__(self, location: np.ndarray, dist_matrix: np.ndarray, num_test: int) -> None:
        super().__init__()
        self.location = location
        self.dist_matrix = dist_matrix
        self.N = dist_matrix.shape[0]
        self.num_test = num_test

        self.origin = 0
        self.path_index = [i for i in range(self.N)]
        self.path_index.remove(self.origin)
        self.iterations = 100  # 迭代次数
        self.improve_time = 10000
        self.popsize = 100  # 种群大小
        self.retain_pr = 0.3  # 强者的定义概率，即种群前30%为强者
        self.random_select_pr = 0.5  # 弱者的存活概率
        self.crossover_pr = 0.95  # 交叉概率
        self.mutation_pr = 0.1  # 变异概率

        self.pathLen = 0
        self.runtime = 0

    def improve(self, code_vector):
        i = 0
        distance = self.cal_fitness([self.origin]+code_vector)
        while i < self.improve_time:
            # randint [a,b]
            u = random.randint(0, self.N-2)
            v = random.randint(0, self.N-2)
            if u != v:
                new_vector = code_vector.copy()
                temp = new_vector[u]
                new_vector[u] = new_vector[v]
                new_vector[v] = temp
                new_distance = self.cal_fitness([self.origin]+new_vector)
                if new_distance < distance:
                    distance = new_distance
                    code_vector = new_vector.copy()
            else:
                continue
            i += 1

        return code_vector

    def init_pop(self):
        # 使用改良圈算法初始化种群
        pop = []
        for i in range(self.popsize):
            # 随机生成个体
            ind = self.path_index.copy()
            random.shuffle(ind)
            ind = self.improve(ind)
            pop.append(ind)

        return pop

    def cal_fitness(self, code_vector):
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

    def selection(self, pop):
        # 对总距离从小到大进行排序
        graded = [[self.cal_fitness([self.origin]+ind), ind] for ind in pop]
        graded = [item[1] for item in sorted(graded)]
        # 选出适应性强的染色体
        retain_length = int(len(graded) * self.retain_pr)
        parents = graded[:retain_length]
        # 选出适应性不强，但是幸存的染色体
        for chromosome in graded[retain_length:]:
            if random.random() < self.random_select_pr:
                parents.append(chromosome)

        return parents

    def crossover(self, parents):
        # 生成子代的个数,以此保证种群稳定
        target_count = self.popsize-len(parents)
        # 孩子列表
        children = []
        while len(children) < target_count:
            male_index = random.randint(0, len(parents) - 1)
            female_index = random.randint(0, len(parents) - 1)
            if male_index != female_index:
                male = parents[male_index]
                female = parents[female_index]

                left = random.randint(0, len(male)-2)
                right = random.randint(left+1, len(male)-1)

                # 交叉片段
                gene1 = male[left:right]
                gene2 = female[left:right]

                child1_c = male[right:]+male[:right]
                child2_c = female[right:]+female[:right]
                child1 = child1_c.copy()
                child2 = child2_c.copy()

                for o in gene2:
                    child1_c.remove(o)

                for o in gene1:
                    child2_c.remove(o)

                child1[left:right] = gene2
                child2[left:right] = gene1

                child1[right:] = child1_c[0:len(child1)-right]
                child1[:left] = child1_c[len(child1)-right:]

                child2[right:] = child2_c[0:len(child1) - right]
                child2[:left] = child2_c[len(child1) - right:]

                children.append(child1)
                children.append(child2)

        return children

    def mutation(self, children):
        for i in range(len(children)):
            if random.random() < self.mutation_pr:
                child = children[i]
                u = random.randint(1, len(child)-4)
                v = random.randint(u+1, len(child)-3)
                w = random.randint(v+1, len(child)-2)
                child = children[i]
                child = child[0:u]+child[v:w]+child[u:v]+child[w:]

        return children

    def get_best_ind(self, pop):
        graded = [[self.cal_fitness([self.origin]+ind), ind]
                  for ind in pop]
        graded = sorted(graded)
        best_fit = graded[0][0]
        best_pop = graded[0][1]

        return best_fit, best_pop

    def search(self) -> List:
        v_dict = dict()

        for i in range(self.num_test):
            print(f"Number {i} test.")

            start_time = time.time()

            iteration = 0
            best_fit_list = []
            pop = self.init_pop()

            best_fit, best_pop = self.get_best_ind(pop)
            # print(f'Fitness of the optimal individual in the initial population: {best_fit:.10f}.')
            best_fit_list.append(best_fit)

            while iteration <= self.iterations:
                # 锦标赛赛选择
                parents = self.selection(pop)
                # 交叉繁殖
                children = self.crossover(parents)
                # 变异操作
                children = self.mutation(children)
                # 更新种群
                pop = parents + children

                best_fit, best_pop = self.get_best_ind(pop)
                best_fit_list.append(best_fit)

                # print(
                #     f'Fitness of the optimal individual in the {iteration}th iteration: {best_fit:%.1f}.')

                iteration += 1
            v = [self.origin] + best_pop + [self.origin]
            path_length = best_fit

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
