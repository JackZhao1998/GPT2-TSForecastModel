# GPT2 Multivariate Time Series Forecasting Model

This is a Time Sereis Forecasting model based on GPT2 structure (decoder-only-based). The model uses naive sinusoid positional encoding to handle both discrete and continous time space.

seed: the random seed used to initialize the training
## model structure hyperparameter
feature_list: a list of strings corresponding to the input features. Example ['x1','x2','x3']
d_input: number of input features
target_var_list: a list of strings corresponding to the output variables, could be univariate or multi-variate. Example ['Y1', 'Y2']
d_output: [int] number of target variables to be predicted
time: [int] a string corresponding to the time columns
d_model: [int] size of hidden layers in the model
num_decoder_lay: [int] number of decoder layers to be used in the model
n_head: [int] numbers of attention head for the model
dropout: [float] proportion of dropout features during training
 
## Training Hyperparameters
n_epoch: [int] numbers of epochs for the training
batch_size: [int] how many samples are feed into the model in each batch

## Optimizer&Scheduler Hyperparameters
learning_rate = 0
betas = (0.9, 0.99)
eps = 1e-9
warmup_steps = 200
factor = 0.2
dropout = 0.2

## path
path: [str] where to save the model state dict. Example: path+'model_name.pt' 
