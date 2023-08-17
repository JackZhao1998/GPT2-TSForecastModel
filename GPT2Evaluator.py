# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 21:28:42 2023

@author: jackz
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from GPT2Model import GPT2Model
from GPT2Model_utils import get_mask, get_tensor, long_loss


def GPT2Evaluater(configs,train_df, val_df, test_df, plot = True, return_details = False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #set seed
    seed = configs.seed

    #define input data
    train_df = train_df
    val_df = val_df
    test_df = test_df

    ## Load model
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = GPT2Model(d_input = configs.d_input, d_output = configs.d_output, d_model=configs.d_model, nhead=configs.n_head,
                        num_decoder_layers=configs.decoder_lay_num).to(device)
    model.load_state_dict(torch.load(configs.path))
    model = model.eval()

     #convert the dataframe to tensor that can be feeded into the model
    val_x, val_obs_time, val_mask, val_y = get_tensor(val_df, feature_list = configs.feature_list, target = configs.target_var_list, obstime = configs.time)

    #divide input and output
    val_x_inp = val_x[:,:-1,:].to(device)
    val_y = val_y[:,1:,:].to(device)
    val_time_inp  = val_obs_time[:,:-1]
    val_time_pred = val_obs_time[:,1:]
    val_mask_inp = get_mask(val_mask[:,:-1].cpu()).to(device)
    val_mask_pred = val_mask[:,1:]

    val_yhat = model(val_x_inp, val_mask_inp, val_time_inp, val_time_pred)
    val_loss = long_loss(val_yhat, val_y, val_mask_pred)
    val_yhat = val_yhat.cpu().detach().numpy().reshape(val_yhat.shape[1])

     #convert the dataframe to tensor that can be feeded into the model
    test_x, test_obs_time, test_mask, test_y = get_tensor(test_df, feature_list = configs.feature_list, target = configs.target_var_list, obstime = configs.time)

    #divide input and output
    test_x_inp = test_x[:,:-1,:].to(device)
    test_y = test_y[:,1:,:].to(device)
    test_time_inp  = test_obs_time[:,:-1]
    test_time_pred = test_obs_time[:,1:]
    test_mask_inp = get_mask(test_mask[:,:-1].cpu()).to(device)
    test_mask_pred = test_mask[:,1:]

    test_yhat = model(test_x_inp, test_mask_inp, test_time_inp, test_time_pred)
    test_loss = long_loss(test_yhat, test_y, test_mask_pred)
    test_yhat = test_yhat.cpu().detach().numpy().reshape(test_yhat.shape[1])
    
    if plot == True: ## modify this part for personal display purpose
        plt.figure()
        t = range(val_yhat.shape[0]-1)
        plt.plot(t,val_yhat[:-1],'b-',label = 'prediction')
        plt.plot(t,val_df['noisy_signal'][1:],'r-',label = 'Signal with Noise')
        plt.plot(t,val_df['signal'][1:],'y-',label = 'Signal')
        plt.title('Validation Prediction Plot')
        plt.legend()
        plt.show()

        plt.figure()
        t = range(test_yhat.shape[0]-1)
        plt.plot(t,test_yhat[:-1],'b-',label = 'prediction')
        plt.plot(t,test_df['noisy_signal'][1:],'r-',label = 'Signal with Noise')
        plt.plot(t,test_df['signal'][1:],'y-',label = 'Signal')
        plt.title('Test Prediction Plot')
        plt.legend()
        plt.show()
        
    if return_details == True:
        return val_loss, test_loss, val_yhat, val_y, test_yhat,  test_y
