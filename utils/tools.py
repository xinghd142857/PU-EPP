# -*- coding: utf-8 -*-
# Author: xinghd


import os
import pandas as pd
import time
import json
import random
import numpy as np

def get_month_day():
    return time.strftime("%m_%d", time.localtime(time.time()))

from gensim.models import Word2Vec
def load_word2vec(path:str):
    assert os.path.exists(path), "Wrong word2vec path"
    return Word2Vec.load(path)

import torch
def d2D(data, device=torch.device('cuda:0')):
    """
    data to device: only for torch.tensor!!!
    device default:torch.device('cuda:0')
    """
    if isinstance (data,(tuple, list)):
        return (d.to(device) for d in data)
    return data.to(device)

def split2PT(df:pd.DataFrame):
    """
    Make sure the order of the columns is [Compound, protein, label(0/1)] !!!
    """
    return df[df.iloc[:,2] > 0.5].reset_index(drop=True), df[df.iloc[:,2] < 0.5].reset_index(drop=True)

def get_data2ind(path_with_filename, *arg):
    """
    if you don't want to save data, you can set path_with_filename=None
    """
    temp = set()
    for a in arg: 
        temp |= set(a)
    temp = {v:i for i, v in enumerate(temp)}
    if path_with_filename:
        with open(path_with_filename, 'w') as f:
            json.dump(temp, f)
    return temp

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True