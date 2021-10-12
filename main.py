#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :main.py
@Description  :
@Date         :2021/10/12 10:20:12
@Author       :Arctic Little Pig
@Version      :1.0
'''

import numpy as np
import pandas as pd

from utils.distance import create_dist_mat
from utils.data_process import create_location


if __name__ == "__main__":
    # np.set_printoptions(threshold=np.inf)

    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 8)

    data_filename = "./data/location.csv"

    _, location = create_location(data_filename)
    dist_matrix = create_dist_mat(location)
    
    print(pd.DataFrame(dist_matrix))
