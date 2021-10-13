#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :name_to_model.py
@Description  :
@Date         :2021/10/13 17:00:40
@Author       :Arctic Little Pig
@Version      :1.0
'''

from models.greedy import Greedy

NAME_TO_MODEL_MAP = {
    "greedy": Greedy,
}


def name2model(model_name: str) -> object:
    model_name = model_name.lower()

    return NAME_TO_MODEL_MAP[model_name]