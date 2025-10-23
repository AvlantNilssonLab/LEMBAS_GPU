"""
Train the signaling model.
"""
from typing import Dict, List, Union
import time

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import wandb
import LEMBAS.utilities as utils
import os

LR_PARAMS = {'max_iter': 5000, 'learning_rate': 2e-3}
OTHER_PARAMS = {'batch_size': 8, 'noise_level': 10, 'gradient_noise_level': 1e-9}
REGULARIZATION_PARAMS = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                   'uniform_max': (1/1.2), 'spectral_loss_factor': 1e-5}
SPECTRAL_RADIUS_PARAMS = {'n_probes_spectral': 5, 'power_steps_spectral': 5, 'subset_n_spectral': 5}
HYPER_PARAMS = {**LR_PARAMS, **OTHER_PARAMS, **REGULARIZATION_PARAMS, **SPECTRAL_RADIUS_PARAMS}


def split_data(X_in: torch.Tensor, 
               y_out: torch.Tensor, 
               train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None}, 
              seed: int = 888):
    """Splits the data into train, test, and validation.

    Parameters
    ----------
    X_in : torch.Tensor
        input ligand concentrations. Index represents samples and columns represent a ligand. Values represent amount of ligand introduced (e.g., concentration). 
    y_out : torch.Tensor
        output TF activities. Index represents samples and columns represent TFs. Values represent activity of the TF.
    train_split_frac : Dict, optional
        fraction of samples to be assigned to each of train, test and split, by default 0.8, 0.2, and 0 respectively
    seed : int, optional
        seed value, by default 888

    """
    
    if not np.isclose(sum([v for v in train_split_frac.values() if v]), 1):
        raise ValueError('Train-test-validation split must sum to 1')
    
    if not train_split_frac['validation'] or train_split_frac['validation'] == 0:
        X_train, X_test, y_train, y_test = train_test_split(X_in, 
                                                        y_out, 
                                                        train_size=train_split_frac['train'],
                                                        random_state=seed,
                                                        shuffle=False)
        X_val, y_val = None, None
    else:
        X_train, _X, y_train, _y = train_test_split(X_in, 
                                                        y_out, 
                                                        train_size=train_split_frac['train'],
                                                        random_state=seed,
                                                        shuffle=False)
        X_test, X_val, y_test, y_val = train_test_split(_X, 
                                                    _y, 
                                                    train_size=train_split_frac['test']/(train_split_frac['test'] + train_split_frac['validation']),
                                                    random_state=seed,
                                                    shuffle=False)

    return X_train, X_test, X_val, y_train, y_test, y_val

class ModelData(Dataset):
    def __init__(self, X_in, y_out):
        self.X_in = X_in
        self.y_out = y_out
    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.X_in.shape[0]
    def __getitem__(self, idx: int):
        "Returns one sample of data, data and label (X, y)."
        return self.X_in[idx, :], self.y_out[idx, :]

def train_signaling_model(mod,config,  
                          optimizer: torch.optim, 
                          loss_fn: torch.nn.modules.loss,
                          reset_epoch : int = 200,
                          hyper_params: Dict[str, Union[int, float]] = None,
                          train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None},
                          train_seed: int = None,
                         verbose: bool = True,
                         dict_to_bench: Dict[str, int] = {"name": "standard", "uniform":0,"L2_or_rand" :0,"device": "cuda"}):
    """Trains the signaling model

    Parameters
    ----------
    mod : SignalingModel
        initialized signaling model. Suggested to also run `mod.signaling_network.prescale_weights` prior to training
    optimizer : torch.optim.adam.Adam
        optimizer to use during training
    loss_fn : torch.nn.modules.loss.MSELoss
        loss function to use during training
    reset_epoch : int, optional
        number of epochs upon which to reset the optimizer state, by default 200
    hyper_params : Dict[str, Union[int, float]], optional
        various hyper parameter inputs for training
            - 'max_iter' : the number of epochs, by default 5000
            - 'learning_rate' : the starting learning rate, by default 2e-3
            - 'batch_size' : number of samples per batch, by default 8
            - 'noise_level' : noise added to signaling network input, by default 10. Set to 0 for no noise. Makes model more robust. 
            - 'gradient_noise_level' : noise added to gradient after backward pass. Makes model more robust. 
            - 'reset_epoch' : number of epochs upon which to reset the optimizer state, by default 200
            - 'param_lambda_L2' : L2 regularization penalty term for most of the model weights and biases
            - 'moa_lambda_L1' : L1 regularization penalty term for incorrect interaction mechanism of action (inhibiting/stimulating)
            - 'ligand_lambda_L2' : L2 regularization penalty term for ligand biases
            - 'uniform_lambda_L2' : L2 regularization penalty term for 
            - 'uniform_max' : 
            - 'spectral_loss_factor' : regularization penalty term for 
            - 'n_probes_spectral' : 
            - 'power_steps_spectral' : 
            - 'subset_n_spectral' : 
    train_split_frac : Dict, optional
        fraction of samples to be assigned to each of train, test and split, by default 0.8, 0.2, and 0 respectively
    train_seed : int, optional
        seed value, by default mod.seed. By explicitly making this an argument, it allows different train-test splits even 
        with the same mod.seed, e.g., for cross-validation
    verbose : bool, optional
        whether to print various progress stats across training epochs


    Returns
    -------
    mod : SignalingModel
        a copy of the input model with trained parameters
    cur_loss : List[float], optional
        a list of the loss (excluding regularizations) across training iterations
    cur_eig : List[float], optional
        a list of the spectral_radius across training iterations
    mean_loss : torch.Tensor
        mean TF activity loss across samples (independent of training)
    X_train : torch.Tensor
        the train split of the input data
    X_test : torch.Tensor
        the test split of the input data
    X_val : torch.Tensor
        the validation split of the input data
    y_train : torch.Tensor
        the train split of the output data
    y_test : torch.Tensor
        the test split of the output data
    y_val : torch.Tensor
        the validation split of the output data
    """
    if not hyper_params:
        hyper_params = HYPER_PARAMS.copy()
    else:
        hyper_params = {k: v for k,v in {**HYPER_PARAMS, **hyper_params}.items() if k in HYPER_PARAMS} # give user input priority
    
    stats = utils.initialize_progress(hyper_params['max_iter'])

    mod = mod.copy() # do not overwrite input
    optimizer = optimizer(mod.parameters(), lr=1, weight_decay=0)
    reset_state = optimizer.state.copy()

    X_in = mod.df_to_tensor(mod.X_in)
    y_out = mod.df_to_tensor(mod.y_out)
    mean_loss = loss_fn(torch.mean(y_out, dim=0) * torch.ones(y_out.shape, device = y_out.device), y_out) # mean TF (across samples) loss
    name_of_run="not_noisy_input_small_batch"
    # set up data objects
    
    if not train_seed:
        train_seed = mod.seed
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X_in, y_out, train_split_frac, train_seed)




    train_data = ModelData(X_train, y_train)
    test_data = ModelData(X_test, y_test)
    if mod.device == 'cuda':
        pin_memory = True
    else:
        pin_memory = False

    # if n_cores != 0:
    #     n_cores_train = min(n_cores, hyper_params['batch_size'])
    # else:
    #     n_cores_train = n_cores
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=1000,#hyper_params['batch_size'],
                                  # num_workers=n_cores_train,
                                  drop_last = False,
                                  #pin_memory = pin_memory,
                                  shuffle=True) 
    test_dataloader = DataLoader(dataset=test_data,
                                  batch_size=1000,#hyper_params['batch_size'],
                                  # num_workers=n_cores_train,
                                  drop_last = False,
                                  #pin_memory = pin_memory,
                                  shuffle=True) 
    start_time = time.time()
    # define what function to use for the benchmarking. There are standard case and non standard case 
    if dict_to_bench["L2_or_rand"]==0:
        L2_reg=mod.L2_reg
    else:
        L2_reg=mod.L2_reg_zero

    if dict_to_bench["uniform"]==0:
        Uni_Reg=mod.uniform_regularization
    else:
        Uni_Reg=mod.moments_based_uniform_regularization
    mean_train=[]#0.0008561546966666355	0.004003495488387461
    mean_test=[]
    with open(f"output_data_{name_of_run}.txt", "w") as file:
        # Example variables: e and cur_loss

        file.write(f"Current e: \t")
        file.write(f"Current train loss:\t")
        file.write(f"Current test loss:\n")
    q=10
    hyper_params['noise_level']=hyper_params['noise_level']*0
    hyper_params['learning_rate']=dict_to_bench['learning_rate']
    hyper_params['lr_final']=dict_to_bench['lr_final']
    # begin iteration
    for e in range(0,hyper_params['max_iter']):
        # set learning rate
        #max_iter: int, max_height: float = 1e-3, 
        #     start_height: float=1e-5, end_height: float=1e-5, 
        #     peak: int = 1000
        cur_lr = utils.get_lr(e, hyper_params['max_iter'], max_height =  1e-3,
                              start_height=1e-5, end_height=1e-5, peak = 1000)
        optimizer.param_groups[0]['lr'] = cur_lr
        
        cur_loss = []
        cur_eig = []
        cur_loss_test=[]
        if e>q+1:
            if (e)%q == 0:
                # Update the file with the new e and last value of cur_loss
                with open(f"output_data_{name_of_run}.txt", "a") as file:
                    file.write(f"{e}\t")
                    file.write(f"{mean_train[-1]}\t")
                    file.write(f"{mean_test[-1]}\n")


        # iterate through batches
        if mod.seed:
            utils.set_seeds(mod.seed + e)
        for batch, (X_in_, y_out_) in enumerate(train_dataloader):
            mod.train()
            optimizer.zero_grad()

            X_in_, y_out_ = X_in_, y_out_
            
            # forward pass
            X_full = mod.input_layer(X_in_) # transform to full network with ligand input concentrations
            utils.set_seeds(mod.seed + mod._gradient_seed_counter)
            network_noise = torch.randn(X_full.shape, device = X_full.device)
            X_full = X_full + (hyper_params['noise_level'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
            Y_full = mod.signaling_network(X_full) # train signaling network weights
            Y_hat = mod.output_layer(Y_full)
            
            # get prediction loss
            fit_loss = loss_fn(y_out_, Y_hat)
            
            # get regularization losses
            sign_reg = mod.signaling_network.sign_regularization(lambda_L1 = hyper_params['moa_lambda_L1']) # incorrect MoA
            ligand_reg = mod.ligand_regularization(lambda_L2 = hyper_params['ligand_lambda_L2']) # ligand biases
            stability_loss, spectral_radius = mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = hyper_params['spectral_loss_factor'],
                                                                                subset_n = hyper_params['subset_n_spectral'], n_probes = hyper_params['n_probes_spectral'], 
                                                                                power_steps = hyper_params['power_steps_spectral'])
            uniform_reg = Uni_Reg(lambda_L2 = hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                     target_min = 0, target_max = hyper_params['uniform_max']) # uniform distribution
            param_reg = L2_reg(hyper_params['param_lambda_L2']) # all model weights and signaling network biases
            
            total_loss = fit_loss + sign_reg + ligand_reg + param_reg + stability_loss + uniform_reg

            total_loss.backward()

            mod.add_gradient_noise(noise_level = hyper_params['gradient_noise_level'])
                
            optimizer.step()

            # store
            cur_eig.append(spectral_radius)
            cur_loss.append(fit_loss.item())
            break
        if e%q==0:

            for batch_test, (X_in_test, y_out_test) in enumerate(test_dataloader):
                mod.train()
                optimizer.zero_grad()

                X_in_test, y_out_test = X_in_test, y_out_test
                
                # forward pass
                X_full = mod.input_layer(X_in_test) # transform to full network with ligand input concentrations
                utils.set_seeds(mod.seed + mod._gradient_seed_counter)
                network_noise = torch.randn(X_full.shape, device = X_full.device)
                X_full = X_full + (hyper_params['noise_level'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
                Y_full = mod.signaling_network(X_full) # train signaling network weights
                Y_hat = mod.output_layer(Y_full)
                
                # get prediction loss
                fit_loss_test = loss_fn(y_out_test, Y_hat)
                cur_loss_test.append(fit_loss_test.item())
                break

            mean_train.append(np.mean(cur_loss))
            mean_test.append(np.mean(cur_loss_test))
            if config!=None:
                wandb.log({"training loss": mean_train[-1]})
                wandb.log({"testing loss": mean_test[-1]})
        if np.logical_and(e % reset_epoch == 0, e>0):
            optimizer.state = reset_state.copy()
        #if e%1000==0:
            # save model
        #    torch.save(mod.state_dict(), f'berzelius_model_{e}_{name_of_run}.pth')

    if verbose:
        mins, secs = divmod(time.time() - start_time, 60)


    return mean_train, mean_test,mod,X_test,y_test