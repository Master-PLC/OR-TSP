#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :main.py
@Description  :
@Date         :2021/10/12 10:20:12
@Author       :Arctic Little Pig
@Version      :1.0
'''

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.distance import create_dist_mat
from utils.data_process import create_location
from utils.name_to_model import name2model


if __name__ == "__main__":
    # np.set_printoptions(threshold=np.inf)

    # pd.set_option('expand_frame_repr', False)
    # pd.set_option('display.max_rows', 10)
    # pd.set_option('display.max_columns', 8)

    parser = argparse.ArgumentParser(description="Solve the TSP Problem")
    parser.add_argument('--data_filename', type=str, default="./data/location.csv",
                        help="Community location file storage location.")
    parser.add_argument('--algo', type=str, default="GA",
                        help="The name of the solution algorithm.")
    parser.add_argument('--image_storage', type=str, default="images",
                        help="The optimal allocation path image storage location obtained by the algorithm.")
    parser.add_argument('--num_test', type=int, default=1,
                        help="The number of tests of the algorithm.")
    config = parser.parse_args()

    _, location = create_location(config.data_filename)
    num_community = location.shape[0]
    dist_matrix = create_dist_mat(location)
    # print(pd.DataFrame(dist_matrix))

    model_type = name2model(config.algo)
    model = model_type(location, dist_matrix, config.num_test)

    path = model.search()
    X, Y = model.get_coordinates(path)
    runtime = model.get_runtime()
    path_length = model.get_path_length()

    print(f"After {config.num_test} tests of {config.algo} algorithm:")
    print("The best material distribution route is: ")
    for i in range(num_community):
        print(f"Community{path[i]} =>", end="\n" if (i+1) % 8 == 0 else " ")
    print(f"Community{path[-1]}")
    print("")
    print(
        f"The shortest distribution distance obtained by solving is: {path_length}.")
    print("")
    print(f"The running time of the program is：{runtime:.12f}s.")

    plt.figure(figsize=(10, 6))
    plt.title(
        f'The optimal distribution path obtained by {config.algo} algorithm')
    plt.plot(X, Y, '-o', ms=6)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    for i, (x, y) in enumerate(zip(X, Y)):
        if i != num_community:
            plt.text(x+0.5, y+0.5, path[i], ha='center', va='bottom', fontsize=8)
    save_path = os.path.join(config.image_storage, f"{config.algo}.png")
    if not os.path.isdir(config.image_storage):
        os.mkdir(config.image_storage)
    plt.savefig(save_path)
    plt.show()
