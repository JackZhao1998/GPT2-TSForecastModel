# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:32:55 2023

@author: jackzs
"""

import numpy as np
import torch
import torch.nn as nn

def get_tensor(df, feature_list = [], target = [], obstime = 'time', device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    '''
    Changes batch data from dataframe to corresponding tensors for Transformer model

    Parameters
    ----------
    df : Pandas Dataframe

    Returns
    -------
    x_long/x_base :
        3d tensor of data with shape (I (subjects), L (visits w/ padding), T (num_features))
    mask :
        2d tensor (1-obs, 0-padding) with shape (I, L)
    obs_time:
        2d tensor of observation times with shape (I, L)
    y:
        target variables to be predicted (I, J, N (N targets))
    '''

    df.loc[:,"id_new"] = df.groupby(by="id").grouper.group_info[0]
    if "visit" not in df:
        visit = df.groupby(by="id").cumcount()
        df.loc[:,"visit"] = visit
    
    #     df["id_new"] = df.groupby("id").grouper.group_info[0]

    #     if "visit" not in df.columns:
    #         df["visit"] = df.groupby("id").cumcount().apply(lambda x: x+1)

    I = len(np.unique(df.loc[:,"id"]))
    max_len = np.max(df.loc[:,"visit"])+2

    x = torch.zeros(I, max_len, len(feature_list))
    y = torch.zeros(I, max_len, len(target))
    mask = torch.zeros((I, max_len), dtype=torch.bool)
    obs_time = torch.zeros(I, max_len)

    for index, row in df.iterrows():

        ii = int(row.loc["id_new"])
        jj = int(row.loc["visit"])

        x[ii,jj,:] = torch.tensor(row.loc[feature_list])
        y[ii,jj,:] = torch.tensor(row.loc[target])

        mask[ii,jj] = 1

        obs_time[ii,jj] = row.loc[obstime]

    return x.to(device), obs_time.to(device), mask.to(device), y.to(device)

def long_loss(yhat, y, mask):
        
    loss_func = nn.MSELoss(reduction='none')
    if y.shape[2] == 1:
        yhat = yhat.reshape(yhat.shape[0],yhat.shape[1])
        y = y.reshape(y.shape[0],yhat.shape[1])
    else:
        mask = mask.repeat((1,1,y.shape[2]))
        
    loss = loss_func(yhat, y)
    loss = loss * mask
    loss = loss.sum()/mask.sum()

    return loss

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

def get_mask(pad = None, future = True, window = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    size = pad.shape[-1]
    mask = (pad != 0).unsqueeze(-2)
    if future:
        future_mask = np.triu(np.ones((1,size,size)), k=1).astype('uint8')==0
        if window is not None:
            win_mask = np.triu(np.ones((1,size,size)), k=-window+1).astype('uint8')==1
            future_mask = future_mask & win_mask
        mask = mask & future_mask
        mask = mask.to(device)
    return mask

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size, warmup, factor):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(optimizer, d_model, warmup_steps=200, factor=1):
    return NoamOpt(optimizer, d_model, warmup_steps, factor)