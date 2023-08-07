# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 21:19:14 2023

@author: jackz
"""
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import GPT2Model.GPT2Model as GPT2Model
from GPT2Model_utils import get_mask, get_std_opt, get_tensor, init_weights, long_loss

def GPT2Trainer(model_configs, train_df, val_df, test_df, plot = True, return_details = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create an instance of the configuration class
    configs = model_configs

    #define input data
    train_df = train_df
    val_df = val_df
    test_df = test_df

    train_id = train_df['id'].unique()

    ## Scale data using Min-Max Scaler
    # transformed_var = ['amplitude0','freq0','phase0','trend']
    # minmax_scaler = MinMaxScaler(feature_range=(-1,1))
    # train_data.loc[:,transformed_var] = minmax_scaler.fit_transform(train_data.loc[:,transformed_var])

    ## Train model
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)

    model = GPT2Model(d_input=configs.d_input, d_output = configs.d_output, d_model=configs.d_model, nhead=configs.n_head,
                        num_decoder_layers=configs.decoder_lay_num).to(device)
    model.apply(init_weights)
    model = model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate, betas=configs.betas, eps=configs.eps)
    scheduler = get_std_opt(optimizer, d_model=configs.d_model, warmup_steps=configs.warmup_steps, factor=configs.factor)

    loss_values = []
    val_loss_values = []
    best_val_loss = np.inf
    start_time = time.time()

    for epoch in range(configs.n_epoch):
        model = model.train()
        for batch in range(0, len(train_id), configs.batch_size):
            optimizer.zero_grad()

            indices = train_id[batch:batch+configs.batch_size]
            batch_df = train_df[train_df["id"].isin(indices)]

            #convert the dataframe to tensor that can be feeded into the model
            batch_x, batch_obs_time, batch_mask, batch_y = get_tensor(batch_df, feature_list = configs.feature_list, target = configs.target_var_list, obstime = 'time')

            #divide input and output
            batch_x_inp = batch_x[:,:-1,:].to(device)
            batch_y = batch_y[:,1:,:].to(device)
            batch_time_inp  = batch_obs_time[:,:-1]
            batch_time_pred = batch_obs_time[:,1:]
            batch_mask_inp = get_mask(batch_mask[:,:-1].cpu()).to(device)
            batch_mask_pred = batch_mask[:,1:]

            batch_yhat = model(batch_x_inp, batch_mask_inp, batch_time_inp, batch_time_pred)

            loss = long_loss(batch_yhat, batch_y, batch_mask_pred)

            loss.backward()
            scheduler.step()


        #------------------------------------------------------------------------------------------------------------------------#

        ## calculate train loss
        # print('--------------------------------------------------')
        # print('Calculating Train Loss')
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        model = model.eval()

        #convert the dataframe to tensor that can be feeded into the model
        train_x, train_obs_time, train_mask, train_y = get_tensor(train_df, feature_list = configs.feature_list, target = configs.target_var_list, obstime = configs.time)

        #divide input and output
        train_x_inp = train_x[:,:-1,:].to(device)
        train_y = train_y[:,1:,:].to(device)
        train_time_inp  = train_obs_time[:,:-1]
        train_time_pred = train_obs_time[:,1:]
        train_mask_inp = get_mask(train_mask[:,:-1].cpu()).to(device)
        train_mask_pred = train_mask[:,1:]

        train_yhat = model(train_x_inp, train_mask_inp, train_time_inp, train_time_pred)
        with torch.no_grad():
            train_loss = long_loss(train_yhat, train_y, train_mask_pred)


        loss_values.append(train_loss.tolist())

        #------------------------------------------------------------------------------------------------------------------------#

        ## calculate validation loss

        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        model = model.eval()

        val_x, val_obs_time, val_mask, val_y = get_tensor(val_df, feature_list = configs.feature_list, target = configs.target_var_list, obstime = configs.time)

        #divide input and output
        val_x_inp = val_x[:,:-1,:].to(device)
        val_y = val_y[:,1:,:].to(device)
        val_time_inp  = val_obs_time[:,:-1]
        val_time_pred = val_obs_time[:,1:]
        val_mask_inp = get_mask(val_mask[:,:-1].cpu()).to(device)
        val_mask_pred = val_mask[:,1:]

        val_yhat = model(val_x_inp, val_mask_inp, val_time_inp, val_time_pred)
        with torch.no_grad():
            val_loss = long_loss(val_yhat, val_y, val_mask_pred)

        val_loss_values.append(val_loss.tolist())


        #------------------------------------------------------------------------------------------------------------------------#


        ## set early stopping

        ## Method 1
        if val_loss_values[-1] < best_val_loss:
            best_val_loss = val_loss_values[-1]
            torch.save(model.state_dict(),configs.path)
            best_epoch = epoch

        if epoch+1 % 5 == 0:
            print(epoch+1,'th epoch finished in', np.round(time.time()-start_time,2),' seconds')
            print('------------------------------------------------------------------')
            
    print("Best epoch is :", best_epoch)
    print("Best val loss is :", best_val_loss)
    print("Training time is", np.round(time.time()-start_time,2), ' seconds.' )
    print('finished')

    if plot == True:
        plt.figure()
        plt.plot(loss_values, 'r-',label='Train')
        plt.plot(val_loss_values, 'b-',label='Val')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot((loss_values-np.min(loss_values))/(np.max(loss_values)-np.min(loss_values)), 'r-',label='Train')
        plt.plot((val_loss_values-np.min(val_loss_values))/(np.max(val_loss_values)-np.min(val_loss_values)), 'b-',label='Test')
        plt.legend()
        plt.show()
    
    if return_details == True:
        return loss_values, val_loss_values