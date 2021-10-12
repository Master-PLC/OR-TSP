#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :main.py
@Description  :
@Date         :2021/10/12 10:20:12
@Author       :Arctic Little Pig
@Version      :1.0
'''

from .utils.data_process import create_location
from .utils.distance import create_dist_mat

data_filename = "./data/location.csv"

location = create_location(data_filename)
dist_matrix = create_dist_mat(location)
print(dist_matrix)
