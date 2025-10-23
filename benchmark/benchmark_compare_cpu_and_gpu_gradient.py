#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_bw, theme, geom_point, geom_hline, annotate, scale_y_log10
import plotnine as p9


# In[2]:


hratch, olof = False, True


# In[3]:


# Dynamically set the absolute path to the LEMBAS directory
if olof:
    sclembas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../LEMBAS'))
elif hratch:
    sclembas_path = os.path.abspath(os.path.join('../../LEMBAS'))

import sys
sys.path.insert(1, sclembas_path)

from LEMBAS.model.bionetwork import format_network, SignalingModel
from LEMBAS.benchmarking_version.benchmark_train import train_signaling_model
import LEMBAS.utilities as utils
from LEMBAS import plotting, io


# In[ ]:


def benchmark_this_dict(dict_to_bench):
    # dict_to_bench is a dictionary that include name,uniform,L2_or_rand,stability_eigen
    # each of these variable control one of the test in benchmark_train. 
    # putting each variable to 0 will imply that the model will run as normal.
    # so 1 is to be read as the indicator for what to test
    # one extra variable is device. this variable control what device we will run our code on

    n_cores = 12
    utils.set_cores(n_cores)

    seed = 888
    if seed:
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        utils.set_seeds(seed = seed)

    
    device = dict_to_bench["device"]
    data_path = '../macrophage_data'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    print("-----------------------------------------------------")
    print("Current device is set to " + device)

    print("-----------------------------------------------------")
    keys_with_one = [key for key, value in dict_to_bench.items() if value == 1]
    if keys_with_one:
        print("Benchmarking being performed:", keys_with_one)
    else:
        print("No relative is being compared.")
    print("-----------------------------------------------------")
    # prior knowledge signaling network
    # Define data_path (adjust the relative path as necessary)


    # Define data_path (adjust the relative path as necessary)

    # Load files using absolute paths
    if olof:

        # Define data_path (adjust the relative path as necessary)
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//macrophage_data'))

        # Load files using absolute paths
        net = pd.read_csv(os.path.join(data_path, 'macrophage-Model.tsv'), sep='\t', index_col=False)
        ligand_input = pd.read_csv(os.path.join(data_path, 'macrophage-Ligands.tsv'), sep='\t', low_memory=False, index_col=0)
        tf_output = pd.read_csv(os.path.join(data_path, 'macrophage-TFs.tsv'), sep='\t', low_memory=False, index_col=0)
    elif hratch:
        data_path = os.path.abspath(os.path.join('..//macrophage_data'))
        net = pd.read_csv('https://zenodo.org/records/10815391/files/macrophage_network.tsv', sep = '\t', index_col = False)
        ligand_input = pd.read_csv('https://zenodo.org/records/10815391/files/macrophage_ligands.tsv', sep = '\t', index_col = 0)
        tf_output = pd.read_csv('https://zenodo.org/records/10815391/files/macrophage_TFs.tsv', sep='\t', low_memory=False, index_col=0)

        
        
    # Defining labels
    stimulation_label = 'stimulation'
    inhibition_label = 'inhibition'
    weight_label = 'mode_of_action'
    source_label = 'source'
    target_label = 'target'
    # Format network
    net = format_network(net, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)

    # Defining network parameters
    # linear scaling of inputs/outputs
    projection_amplitude_in = 3
    projection_amplitude_out = 1.2
    # other parameters
    bionet_params = {'target_steps': 5, 'max_steps': 100, 'exp_factor':50, 'tolerance': 1e-20, 'leak':1e-2} # fed directly to model

    # training parameters
    lr_params = {'max_iter': 10, 
                'learning_rate': 2e-3}
    other_params = {'batch_size': 10, 'noise_level': 10, 'gradient_noise_level': 1e-9}
    regularization_params = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                    'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 0*1e-5}
    spectral_radius_params = {'n_probes_spectral': 3, 'power_steps_spectral': 3, 'subset_n_spectral': 3}
    target_spectral_radius = 0.8
    hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params} # fed into training function


    # Define the network
    mod = SignalingModel(net = net,
                        X_in = ligand_input,
                        y_out = tf_output, 
                        projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                        weight_label = weight_label, source_label = source_label, target_label = target_label,
                        bionet_params = bionet_params, 
                        dtype = torch.float32, device = device, seed = seed)




    X_in = mod.df_to_tensor(mod.X_in)
    y_out = mod.df_to_tensor(mod.y_out)
    # model setup
    mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
    mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

    # loss and optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam
    print("Training has started")
    print("-----------------------------------------------------")
    print("Printing training progress")
    print("-----------------------------------------------------")


    # training loop
    mod, cur_loss, cur_eig, mean_loss, stats, X_train, X_test, X_val, y_train, y_test, y_val = train_signaling_model(mod, optimizer, loss_fn, 
                                                                    reset_epoch = 200, hyper_params = hyper_params, 
                                                                    train_seed = seed, verbose = True ,dict_to_bench=dict_to_bench) 
    cur_eig
    # store results
    plt.plot(cur_eig)
    plt.show()
    data_path='LEMBAS/benchmark/models'

    os.makedirs(data_path, exist_ok=True)  # Create directory if it doesn't exist

    #data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//macrophage_data'))
    io.write_pickled_object(stats, os.path.join(data_path, dict_to_bench["name"]+'_training_stats.pickle'))
    torch.save(obj=mod.state_dict(), f=os.path.join(data_path, dict_to_bench["name"]+'_mac_state_dict.pth'))

    # Load relevant models and states
    mod.load_state_dict(torch.load(os.path.join(data_path, dict_to_bench["name"]+'_mac_state_dict.pth')))
    stats = io.read_pickled_object(os.path.join(data_path, dict_to_bench["name"]+ '_training_stats.pickle'))

    # Prepare model to be evaluated
    mod.eval()
    Y_hat, Y_full = mod(X_in)
    Y_full
    # save Y_hat
    io.write_pickled_object(Y_hat, os.path.join(data_path, dict_to_bench["name"]+'_Y_hat.pickle'))
    mod.zero_grad()  # This resets all gradients to zero
    mod.train()
    Y_hat, Y_full = mod(X_in)
    fit_loss = loss_fn(y_out, Y_hat)
    path= mod.signaling_network.forward_path(mod.input_layer(X_in))[1]
    print(fit_loss.item())# print all values off loss
    # Select random features and samples
    plt.figure(figsize=(10, 6))
    for i in range(0,10):
        random_feature = torch.randint(0, path.shape[2], (1,)).item()  # Random feature index
        random_sample = torch.randint(0, path.shape[1], (1,)).item()   # Random sample index

        # Extract the data for the selected feature and sample across all time steps
        time_series = path[:, random_sample, random_feature]  # Shape: [100]
        plt.plot(time_series.detach().cpu().numpy(), label=f'Feature {random_feature}, Sample {random_sample}')
    # Plot the time series

   
    plt.xlabel('Time Steps')
    plt.ylabel('Feature Value')
    plt.title(f'Time Series of Feature {random_feature} for Sample {random_sample}')
    plt.legend()
    plt.grid(True)
    plt.show()
    fit_loss.backward()
    gradient_vector = mod.signaling_network.weights.grad[mod.signaling_network.weights!=0]
    io.write_pickled_object(gradient_vector, os.path.join(data_path, dict_to_bench["name"]+'_gradient_vector.pickle'))
    gradient_vector = mod.signaling_network.bias.grad
    io.write_pickled_object(gradient_vector, os.path.join(data_path, dict_to_bench["name"]+'_gradient_bias.pickle'))
    # Define output directory
    # Reset gradients before the next backward pass
    mod.zero_grad()  # This resets all gradients to zero

    # Prepare model to be evaluated
    mod.eval()
    Y_hat, Y_full = mod.forward_for_gpu_cpu_compare(X_in)
    # save Y_hat
    io.write_pickled_object(Y_hat, os.path.join(data_path, dict_to_bench["name"]+'_Y_hat_detach_and_100_steps.pickle'))
    mod.train()
    Y_hat, Y_full = mod(X_in)
    fit_loss = loss_fn(y_out, Y_hat)
    print(fit_loss.item())# print all values off loss

    fit_loss.backward()
    gradient_vector = mod.signaling_network.weights.grad[mod.signaling_network.weights!=0]
    io.write_pickled_object(gradient_vector, os.path.join(data_path, dict_to_bench["name"]+'_gradient_vector_detach_and_100_steps.pickle'))
    gradient_vector = mod.signaling_network.bias.grad
    io.write_pickled_object(gradient_vector, os.path.join(data_path, dict_to_bench["name"]+'_gradient_bias_detach_and_100_steps.pickle'))
    # Define output directory

# 
dict_to_bench={"name": "model_to_test_on_CPU_and_GPU", "uniform":1,"L2_or_rand" :1,"device": "cuda","test_time":1, 
              'ProjectOutput_bias': 0}
                          
benchmark_this_dict(dict_to_bench)

