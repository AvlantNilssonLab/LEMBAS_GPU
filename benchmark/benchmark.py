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
    bionet_params = {'target_steps': 10, 'max_steps': 10, 'exp_factor':50, 'tolerance': 1e-20, 'leak':1e-2} # fed directly to model

    # training parameters
    lr_params = {'max_iter': 5000, 
                'learning_rate': 2e-3}
    other_params = {'batch_size': 10, 'noise_level': 10, 'gradient_noise_level': 1e-9}
    regularization_params = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                    'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-5}
    spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 10, 'subset_n_spectral': 10}
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

    # store results
    data_path='LEMBAS/benchmark/models'
    #data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//macrophage_data'))
    io.write_pickled_object(stats, os.path.join(data_path, dict_to_bench["name"]+'_training_stats.pickle'))
    torch.save(obj=mod.state_dict(), f=os.path.join(data_path, dict_to_bench["name"]+'_mac_state_dict.pth'))

    # Load relevant models and states
    mod.load_state_dict(torch.load(os.path.join(data_path, dict_to_bench["name"]+'_mac_state_dict.pth')))
    stats = io.read_pickled_object(os.path.join(data_path, dict_to_bench["name"]+ '_training_stats.pickle'))

    # Prepare model to be evaluated
    mod.eval()
    Y_hat, Y_full = mod(X_in)
    # save Y_hat
    io.write_pickled_object(Y_hat, os.path.join(data_path, dict_to_bench["name"]+'_Y_hat.pickle'))
    mod.train()
    Y_hat, Y_full = mod(X_in)
    fit_loss = loss_fn(y_out, Y_hat)
    print(fit_loss.item())# print all values off loss

    fit_loss.backward()
    gradient_vector = mod.signaling_network.weights.grad[mod.signaling_network.weights!=0]
    io.write_pickled_object(gradient_vector, os.path.join(data_path, dict_to_bench["name"]+'_gradient_vector.pickle'))
    gradient_vector = mod.signaling_network.bias.grad
    io.write_pickled_object(gradient_vector, os.path.join(data_path, dict_to_bench["name"]+'_gradient_bias.pickle'))
    # Define output directory
    output_dir = "LEMBAS/benchmark/figures"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # --- Figure 1: Loss over time ---
    loss_smooth = utils.get_moving_average(values=stats['loss_mean'], n_steps=5)
    loss_sigma_smooth = utils.get_moving_average(values=stats['loss_sigma'], n_steps=10)
    epochs = np.arange(len(stats['loss_mean']))

    p1A = plotting.shade_plot(X=epochs, Y=loss_smooth, sigma=loss_sigma_smooth, x_label='Epoch', y_label='Loss')
    p1A += p9.scale_y_log10()
    p1A += p9.geom_hline(yintercept=mean_loss.item(), linetype="dashed", color="black")
    p1A.save(f"{output_dir}/{dict_to_bench["name"]}loss_over_time.png", dpi=300, verbose = False)  # Save the plot

    # --- Figure 2: Learning Rate over time ---
    viz_df = pd.DataFrame(data={'Epoch': epochs, 'lr': stats['learning_rate']})
    width, height = 5, 3
    p1B = (
        p9.ggplot(viz_df, p9.aes(x='Epoch', y='lr')) +
        p9.geom_line(color='#1E90FF') +
        p9.theme_bw() + 
        p9.theme(figure_size=(width, height)) +
        p9.ylab('Learning Rate')
    )
    p1B.save(f"{output_dir}/{dict_to_bench["name"]}learning_rate_over_time.png", dpi=300, verbose = False)  # Save the plot

    # --- Figure 3: Spectral Radius ---
    eig_smooth = utils.get_moving_average(stats['eig_mean'], 5)
    eig_sigma_smooth = utils.get_moving_average(stats['eig_sigma'], 5)

    p1C = plotting.shade_plot(X=epochs, Y=eig_smooth, sigma=eig_sigma_smooth, x_label='Epoch', y_label='Spectral Radius')
    p1C += p9.geom_hline(yintercept=mod.signaling_network.training_params['spectral_target'], linetype="dashed", color="black")
    p1C += p9.geom_hline(yintercept=1, color="black")
    p1C.save(f"{output_dir}/{dict_to_bench["name"]}spectral_radius_over_time.png", dpi=300, verbose = False)  # Save the plot

    # --- Figure 4: Prediction vs Actual ---
    y_pred = Y_hat.detach().flatten().cpu().numpy()
    y_actual = y_out.detach().flatten().cpu().numpy()
    pr, _ = pearsonr(y_pred, y_actual)

    viz_df = pd.DataFrame(data={'Predicted': y_pred, 'Actual': y_actual})
    width, height = 3, 3
    p2 = (
        p9.ggplot() +
        p9.geom_point(data=viz_df, mapping=p9.aes(x='Predicted', y='Actual'), color='#1E90FF') +
        p9.geom_line(data=pd.DataFrame(data={'x': [0, 1], 'y': [0, 1]}), mapping=p9.aes(x='x', y='y'), color='black') +
        p9.theme_bw() + 
        p9.theme(figure_size=(width, height)) +
        p9.annotate(geom='text', x=0.25, y=0.95, label='Pearson r: {:.2f}'.format(pr), size=10)
    )
    p2.save(f"{output_dir}/{dict_to_bench["name"]}predicted_vs_actual_unif_.png", dpi=300, verbose = False)
    # Save the raw prediction and actual data to a CSV file
    raw_data = pd.DataFrame({'Predicted': y_pred, 'Actual': y_actual})  # {{ edit_1 }}
    raw_data.to_csv(f"LEMBAS/benchmark/data_generated/{dict_to_bench['name']}_predicted_vs_actual_data.csv", index=False)  # {{ edit_2 }}
    # --- Figure 5: Control Distance vs Fit Distance ---
    X_ctrl = torch.zeros((1, X_in.shape[1]), dtype=mod.dtype, device = mod.device)
    Y_ctrl, _ = mod(X_ctrl)
    signal_distance = torch.sum(torch.abs(Y_hat - Y_ctrl), dim=1).detach().cpu().numpy().flatten()
    fit_distance = torch.sum(torch.square(Y_hat - y_out), dim=1).detach().cpu().numpy().flatten()

    viz_df = pd.DataFrame(data={'Control Distance': signal_distance, 'Fit Distance': fit_distance})
    width, height = 3, 3
    p3 = (
        p9.ggplot() +
        p9.geom_point(data=viz_df, mapping=p9.aes(x='Control Distance', y='Fit Distance'), color='#1E90FF') +
        p9.theme_bw() + 
        p9.theme(figure_size=(width, height))
    )
    p3.save(f"{output_dir}/{dict_to_bench["name"]}Control_Distance_.png", dpi=300, verbose = False)

    print("-----------------------------------------------------")
    print("training is complete and figures and data are saved")
    print("-----------------------------------------------------")


# 
dict_to_bench={"name": "model_to_test_on_CPU_and_GPU", "uniform":1,"L2_or_rand" :1,"device": "cuda","test_time":1, 
              'ProjectOutput_bias': 0}
                          
benchmark_this_dict(dict_to_bench)

