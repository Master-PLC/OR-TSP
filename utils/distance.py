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


def create_dist_mat(location: pd.DataFrame) -> np.ndarray:
    """
    Description::

    :param :
    :return :

    Usage::

    """

    N, _ = location.shape

    dist_matrix = np.zeros([N, N])

    for i in range(N):
        vi = location.loc[i+1, ["x", "y"]].values
        for j in range(N):
            if j != i:
                vj = location.loc[j+1, ["x", "y"]].values
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
