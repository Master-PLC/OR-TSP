#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :seed_init.py
@Description  :
@Date         :2021/10/17 16:51:00
@Author       :Arctic Little Pig
@Version      :1.0
'''

import random

import numpy as np
import torch

seed = 58888

def init_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
