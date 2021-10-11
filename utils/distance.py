#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :distance.py
@Description  :
@Date         :2021/10/11 18:32:03
@Author       :Arctic Little Pig
@Version      :1.0
'''

import pandas as pd
import numpy as np

def create_dist_mat(location:pd.DataFrame)->np.ndarray:
    