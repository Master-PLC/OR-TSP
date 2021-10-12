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
from typing import Union

from pandas.core.algorithms import isin


def create_dist_mat(location: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Description::

    :param :
    :return :

    Usage::

    """
    if isinstance(location, pd.DataFrame):
        location = location.loc[:, ["x", "y"]].values
    
    N, _ = location.shape

    dist_matrix = np.zeros([N, N])

    for i in range(N):
        vi = location[i, :]
        for j in range(N):
            if j != i:
                vj = location[j, :]
                dij = np.sqrt(np.sum((vi - vj)**2))
                dist_matrix[i, j] = dij

    return dist_matrix


if __name__ == "__main__":
    num_point = 3
    location_dict = {"x": list(range(1, num_point+1)),
                     "y": list(range(1, num_point+1))}
    location = pd.DataFrame(location_dict, pd.Index(range(1, num_point+1)))
    # print(location)

    dist_matrix = create_dist_mat(location)
    print(dist_matrix)
