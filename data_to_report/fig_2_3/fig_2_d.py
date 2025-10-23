import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_bw, theme, geom_point, geom_hline, annotate, scale_y_log10
import plotnine as p9
import tqdm as tqdm
import seaborn as sns  # {{ edit_1 }}
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Dynamically set the absolute path to the LEMBAS directory
sclembas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

import sys
sys.path.insert(1, sclembas_path)

from LEMBAS.model.bionetwork import format_network, SignalingModel
import LEMBAS.utilities as utils
from LEMBAS import plotting, io
# %% 


def self_prune_test(dict_to_bench):
    # dict_to_bench is a dictionary that include name,uniform,L2_or_rand,test_eigen_1_or_uni_0
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
    # Defining labels
    stimulation_label = 'stimulation'
    inhibition_label = 'inhibition'
    weight_label = 'mode_of_action'
    source_label = 'source'
    target_label = 'target'
    # Format network
    data_set=dict_to_bench["data_set"]
    
    if data_set == "macrophage":
        
        ligand_input = pd.read_csv(('LEMBAS/macrophage_data/'+ 'macrophage-Ligands.tsv'), sep='\t', low_memory=False, index_col=0)
        tf_output = pd.read_csv(('LEMBAS/macrophage_data/'+ 'macrophage-TFs.tsv'), sep='\t', low_memory=False, index_col=0)
        net = pd.read_csv(('LEMBAS/macrophage_data/'+ 'macrophage-Model.tsv'), sep='\t', index_col=False)
        net = format_network(net, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)

    elif data_set == "ligand":
        
        ligand_input = pd.read_csv(('LEMBAS/ligand_data/'+ 'ligandScreen-Ligands.tsv'), sep='\t', low_memory=False, index_col=0)
        tf_output = pd.read_csv(('LEMBAS/ligand_data/'+ 'ligandScreen-TFs.tsv'), sep='\t', low_memory=False, index_col=0)
        net = pd.read_csv(('LEMBAS/ligand_data/'+ 'ligandScreen-Model.tsv'), sep='\t', index_col=False)
        net = format_network(net, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)

    else:
        net = pd.read_csv(('LEMBAS/synthetic_data/'+ 'KEGGnet-Model.tsv'), sep='\t', index_col=False)
        net = format_network(net, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)
        annotation = pd.read_csv('LEMBAS/synthetic_data/'+ 'KEGGnet-Annotation.tsv', sep='\t')
        ligand_input = torch.load('LEMBAS/synthetic_data/X.pt')
        tf_output = torch.load('LEMBAS/synthetic_data/Y.pt')

        inName = annotation.loc[annotation['ligand'],'code'].values
        outName = annotation.loc[annotation['TF'],'code'].values

        # If X_synthetic and Y_synthetic are torch tensors, convert them to pandas DataFrames
        X_synthetic_df = pd.DataFrame(ligand_input.numpy())
        Y_synthetic_df = pd.DataFrame(tf_output.numpy())

        # Update column names for X and Y
        inName = np.intersect1d(net['source'].values, inName)
        outName = np.intersect1d(net['target'].values, outName)

        X_synthetic_df.columns = inName  # Set column names for X_synthetic
        Y_synthetic_df.columns = outName  # Set column names for Y_synthetic

        # Set row names starting from zero for both
        X_synthetic_df.index = np.arange(len(X_synthetic_df))
        Y_synthetic_df.index = np.arange(len(Y_synthetic_df))

        ligand_input = X_synthetic_df
        tf_output = Y_synthetic_df

    
    

    # Defining network parameters
    # linear scaling of inputs/outputs
    projection_amplitude_in = 3
    projection_amplitude_out = 1.2
    # other parameters
    bionet_params = {'target_steps': 100, 'max_steps': 150, 'exp_factor':21, 'tolerance': 1e-5, 'leak':1e-2} # fed directly to model


    other_params = {'batch_size': 8, 'noise_level': 10, 'gradient_noise_level': 1e-9}
    regularization_params = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                    'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-5}
    spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 5, 'subset_n_spectral': 10}
    target_spectral_radius = 0.8
    


    # Plot with removed weight sizes on x-axis
    plt.figure(figsize=(5,5))

    lr_params = {'max_iter':1, 
                'learning_rate': 2e-3}
    spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 1, 'subset_n_spectral': 10}
    hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params} # fed into training function
    # Define the network
    mod = SignalingModel(net = net,
                            X_in = ligand_input,
                            y_out = tf_output, 
                            projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                            weight_label = weight_label, source_label = source_label, target_label = target_label,
                            bionet_params = bionet_params, 
                            dtype = torch.float32, device = "cuda", seed = seed)

    ligand_input=torch.tensor(ligand_input.values, dtype = torch.float32).to('cuda')
    tf_output=torch.tensor(tf_output.values, dtype = torch.float32).to('cuda')
    removed_weight_sizes_all = []
    all_MAE_values = []
    for q in tqdm.tqdm(range(0,50)):
        if data_set == "macrophage":
            state_dict_path = f'LEMBAS/data_to_report/fig_2_3/parameter_study_mac_{dict_to_bench["L2"]}/{q}_model_2499.pth'
        elif data_set == "ligand":
            state_dict_path = f'LEMBAS/data_to_report/fig_2_3/parameter_study_ligand_{dict_to_bench["L2"]}/{q}_model_7999.pth'
        else:
            state_dict_path = f'LEMBAS/data_to_report/fig_2_3/parameter_study_syn_{dict_to_bench["L2"]}/{q}_model_2499.pth'
        #state_dict = torch.load(state_dict_path, 
        #                            map_location=torch.device('cpu'))

        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        mod.load_state_dict(state_dict, strict = False)
        mod=mod.to('cuda')


        # Copy weights so we don’t modify the original permanently
        weights = mod.signaling_network.weights.data.clone()

        # Flatten and find nonzero weights
        flat_weights = weights.view(-1)
        nonzero_indices = torch.where(flat_weights != 0)[0]

        # Sort nonzero weights by absolute value
        sorted_indices = nonzero_indices[torch.argsort(flat_weights[nonzero_indices].abs())]
        if data_set == "synthetic":
            step_size = 10
        else:
            step_size = 500
        y_results = []
        removed_weight_sizes = []  # <-- NEW: track weight magnitudes

        for i in (range(0, len(sorted_indices), step_size)):
            indices_to_zero = sorted_indices[i:i+step_size]

            # Track the average absolute weight being removed
            max_weight = flat_weights[indices_to_zero].abs().max().item()
            removed_weight_sizes.append(max_weight)

            # Zero the selected weights
            weights.view(-1)[indices_to_zero] = 0.0

            # Apply updated weights to model
            mod.signaling_network.weights.data = weights

            # Forward pass
            y, _ = mod(ligand_input)
            

            # Store output
            y_results.append(y.clone().detach())
        removed_weight_sizes_all.append(removed_weight_sizes)
        baseline_y = y_results[0].view(-1).cpu().numpy()

        
        MAE_values = [
            mean_absolute_error(baseline_y, y.view(-1).cpu().numpy())
            for y in (y_results)
        ]
        all_MAE_values.append(MAE_values)  # <-- store for mean
        plt.plot(removed_weight_sizes, MAE_values, color="blue", linewidth=2.0,alpha=0.1)
    removed_weight_sizes_all_flatten = np.array(removed_weight_sizes_all).flatten()

    # Add red vertical lines at 25%, 50%, and 75% of edges removed
    total_edges = len(sorted_indices)
    # same as in pruning loop
    num_steps = len(removed_weight_sizes)

    for perc in [0.25, 0.50, 0.75]:  # 25%, 50%, 75%
        edges_removed_threshold = perc * total_edges
        step_index = int(edges_removed_threshold / step_size)
        if step_index < num_steps:
            plt.axvline(x=removed_weight_sizes_all_flatten[step_index], color="red", linestyle="--", alpha=0.7)

            # Get current y-axis limits to center label
            ymin, ymax = plt.ylim()
            ymid = (ymin + ymax) / 2

            plt.text(
                removed_weight_sizes_all_flatten[step_index]*1.1, ymid, 
                f"{int(perc*100)}%", color="red", 
                rotation=90, verticalalignment="center",fontsize=14
            )

    plt.xlabel("Removed Weight Sizes",fontsize=14)
    plt.xscale('log')
    plt.ylabel("MAE",fontsize=14)
    
    if dict_to_bench["L2"] == 1:
        L2_label = "L2 low"
    elif dict_to_bench["L2"] == 10: 
        L2_label = "L2 medium"
    elif dict_to_bench["L2"] == 100:
        L2_label = "L2 high"
    else:
        L2_label = "L2"

    if data_set == "macrophage":
        plt.title("Low-Coverage"+", "+L2_label ,fontsize=14)
    elif data_set == "ligand":
        plt.title("High-Coverage"+", "+L2_label ,fontsize=14)
    else:
        plt.title("Synthetic"+", "+L2_label ,fontsize=14)
    # Convert to numpy array
    all_MAE_values = np.array(all_MAE_values)
    removed_weight_sizes_all= np.array(removed_weight_sizes_all)

    # Compute mean MAE across runs at each removal step
    mean_MAE = all_MAE_values.mean(axis=0)
    mean_removed = removed_weight_sizes_all.mean(axis=0)

    # Plot mean path as black line
    plt.plot(mean_removed, mean_MAE, color="black", linewidth=2.5, label="Mean MAE")
    plt.legend(loc="upper left")
    #plt.grid(True)
    #plt.legend()
    print(dict_to_bench["L2"])
    plt.xlim(10**-3, 2)
    plt.ylim(0.0)
    plt.savefig(f'LEMBAS/data_to_report/figs/Stability_test_weight_size_{data_set}_L2_{dict_to_bench["L2"]}.svg', dpi=300, bbox_inches='tight')



data_sets = ["ligand","macrophage",  "synthetic"]
L2s = [1, 10, 100]
dict_to_bench={"name": "power_iterations", "L2_or_rand" :0,"test_eigen_1_or_uni_0":0,"device": "cuda","data_set": "macrophage","L2" :10}

for data_set in data_sets:
    for L2 in L2s:
        dict_to_bench["L2"] = L2
        dict_to_bench["data_set"] = data_set
        self_prune_test(dict_to_bench)

