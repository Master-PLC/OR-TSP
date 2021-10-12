#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :data_process.py
@Description  :
@Date         :2021/10/11 16:19:48
@Author       :Arctic Little Pig
@Version      :1.0
'''
import os
import pandas as pd


DATA_DIR = "../data"
INDEX_COL = "No"


def create_location(file_path: str) -> pd.DataFrame:
    """
    Description::用于读取社区位置的csv文件
    
    :param file_path:社区位置的csv文件路径
    :return df:返回包含社区序号、社区名称以及社区坐标的DataFrame
    
    Usage::
    
    """
    
    df = pd.read_csv(file_path,
                     index_col=INDEX_COL,
                     header=0,
                     names=['No', 'Name', 'x', 'y'])

    return df


if __name__ == "__main__":
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)

    file_name = "location.csv"
    file_path = os.path.join(DATA_DIR, file_name)

    location_table = create_location(file_path)

    # print(location_table)
    print(location_table.loc[:, ["x", "y"]])
    # print(len(location_table))
    # print(location_table.columns)
    print(location_table.shape)
    # print(location_table.loc[1, ["x", "y"]].values)
    # print(location_table.loc[:16, ["x", "y"]])
    # print(location_table.loc[17:, ["x", "y"]])
