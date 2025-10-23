import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_bw, theme, geom_point, geom_hline, annotate, scale_y_log10
import plotnine as p9
import tqdm as tqdm


# Dynamically set the absolute path to the LEMBAS directory
sclembas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../LEMBAS'))

import sys
sys.path.insert(1, sclembas_path)

from LEMBAS.benchmarking_version.self_prune_bionetwork import format_network, SignalingModel
from LEMBAS.benchmarking_version.self_prune_train_figure import train_signaling_model
import LEMBAS.utilities as utils
from LEMBAS import plotting, io
# %% 


def benchmark_this_dict(dict_to_bench):
    # dict_to_bench is a dictionary that include name,uniform,L2_or_rand,stability_eigen
    # each of these variable control one of the test in benchmark_train. 
    # putting each variable to 0 will imply that the model will run as normal.
    # so 1 is to be read as the indicator for what to test
    # one extra variable is device. this variable control what device we will run our code on

    # dict_to_bench is a dictionary that include name,uniform,L2_or_rand,stability_eigen
    # each of these variable control one of the test in benchmark_train. 
    # putting each variable to 0 will imply that the model will run as normal.
    # so 1 is to be read as the indicator for what to test
    # one extra variable is device. this variable control what device we will run our code on

    n_cores = 12
    utils.set_cores(n_cores)

    #if seed:
    #    torch.use_deterministic_algorithms(True)
    #    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #    utils.set_seeds(seed = seed)
    import random
    #seed=random.randint(0,1000)
    seed=dict_to_bench["seed"]
    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    device = dict_to_bench["device"]
    data_path = '../synthetic_data'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    keys_with_one = [key for key, value in dict_to_bench.items() if value == 1]

    # prior knowledge signaling network
    # Define data_path (adjust the relative path as necessary)



    # Defining labels
    stimulation_label = 'stimulation'
    inhibition_label = 'inhibition'
    weight_label = 'mode_of_action'
    source_label = 'source'
    target_label = 'target'
    # Format network

    


    data_set=1
    if data_set==0:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//synthetic_data'))


        networkList = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//synthetic_data/KEGGnet-Model.tsv')), sep='\t', index_col=False) 
        networkList = format_network(networkList, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)

        annotation = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//synthetic_data/KEGGnet-Annotation.tsv')), sep='\t')
        X_synthetic = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//synthetic_data/X_10000.pt')))
        Y_synthetic = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//synthetic_data/Y_10000.pt')))
        

        X_synthetic_test = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//synthetic_data/X_test.pt')))
        Y_synthetic_test = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//synthetic_data/Y_test.pt')))

        X_combined = torch.cat([X_synthetic, X_synthetic_test], dim=0)
        Y_combined = torch.cat([Y_synthetic, Y_synthetic_test], dim=0)
        # Get the total number of samples
        #total_samples = X_combined.size(0)

        # Generate a random permutation of indices
        indices = torch.randperm(10000)

        # Select the first 'global_num' indices for the random subset
        selected_indices = indices[:global_num]
        # Split the data based on the selected indices
        X_synthetic = X_combined[selected_indices]
        Y_synthetic = Y_combined[selected_indices]
        # Concatenate training and test data


        inName = annotation.loc[annotation['ligand'],'code'].values
        outName = annotation.loc[annotation['TF'],'code'].values

        # If X_synthetic and Y_synthetic are torch tensors, convert them to pandas DataFrames
        X_synthetic_df = pd.DataFrame(X_synthetic.numpy())
        Y_synthetic_df = pd.DataFrame(Y_synthetic.numpy())

        # Update column names for X and Y
        inName = np.intersect1d(networkList['source'].values, inName)
        outName = np.intersect1d(networkList['target'].values, outName)

        X_synthetic_df.columns = inName  # Set column names for X_synthetic
        Y_synthetic_df.columns = outName  # Set column names for Y_synthetic

        # Set row names starting from zero for both
        X_synthetic_df.index = np.arange(len(X_synthetic_df))
        Y_synthetic_df.index = np.arange(len(Y_synthetic_df))

        ligand_input = X_synthetic_df
        tf_output = Y_synthetic_df
    elif data_set==1:
            # Define data_path (adjust the relative path as necessary)
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//macrophage_data'))

        # Load files using absolute paths
        networkList = pd.read_csv(os.path.join(data_path, 'macrophage-Model.tsv'), sep='\t', index_col=False)
        ligand_input = pd.read_csv(os.path.join(data_path, 'macrophage-Ligands.tsv'), sep='\t', low_memory=False, index_col=0)
        tf_output = pd.read_csv(os.path.join(data_path, 'macrophage-TFs.tsv'), sep='\t', low_memory=False, index_col=0)
        networkList = format_network(networkList, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)
        X_combined=torch.tensor(ligand_input.values)
        Y_combined=torch.tensor(tf_output.values)
        # Generate a random permutation of indices

        # Split the data based on the selected indices
        X_synthetic = X_combined
        Y_synthetic = Y_combined
    else:
            # Define data_path (adjust the relative path as necessary)
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//ligand_data'))

        # Load files using absolute paths
        networkList = pd.read_csv(os.path.join(data_path, 'ligandScreen-Model.tsv'), sep='\t', index_col=False)
        ligand_input = pd.read_csv(os.path.join(data_path, 'ligandScreen-Ligands.tsv'), sep='\t', low_memory=False, index_col=0)
        tf_output = pd.read_csv(os.path.join(data_path, 'ligandScreen-TFs.tsv'), sep='\t', low_memory=False, index_col=0)
        networkList = format_network(networkList, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)
        X_combined=torch.tensor(ligand_input.values)
        Y_combined=torch.tensor(tf_output.values)
        # Generate a random permutation of indices

        # Split the data based on the selected indices
        X_synthetic = X_combined
        Y_synthetic = Y_combined

    # Construct the folder name
    folder_name = f'ligand_first_test/sample_size_{dict_to_bench["batch"]}_{dict_to_bench["edges_to_add"]}'

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)


    # Defining network parameters
    # linear scaling of inputs/outputs

    projection_amplitude_in = 3
    projection_amplitude_out = 1.2
    # other parameters
    bionet_params = {'target_steps': 150, 'max_steps': 150, 'exp_factor':50, 'tolerance': 1e-5, 'leak':1e-2} # fed directly to model
    # training parameters
    lr_params = {'max_iter': 5000, 
                'learning_rate': 2e-3}
    other_params = {'batch_size': 10, 'noise_level': 10, 'gradient_noise_level': 1e-9}
    regularization_params = {'param_lambda_L2': 1e-6*dict_to_bench["L2_norm"], 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                    'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-5}
    spectral_radius_params = {'n_probes_spectral': 2, 'power_steps_spectral': 5, 'subset_n_spectral': 2}
    target_spectral_radius = 0.9
    hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params} # fed into training function


    # Define the network
    mod = SignalingModel(net = networkList,
                        X_in = ligand_input,
                        y_out = tf_output, 
                        projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                        weight_label = weight_label, source_label = source_label, target_label = target_label,
                        bionet_params = bionet_params, 
                        dtype = torch.float32, device = device, seed=seed,dict_to_bench=dict_to_bench)


    # transform dataframe to tensors
    #X_in = mod.df_to_tensor(mod.X_synthetic)
    #y_out = mod.df_to_tensor(mod.Y_synthetic)



    # model setup
    mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
    mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

    # loss and optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam


    mean_train, mean_test,mod,X_test,y_test,true_net,false_net,true_per,stocastic_per,correlator_over_time,correlator_over_time_true = train_signaling_model(mod, optimizer, loss_fn, 
                                                                    reset_epoch = 200, hyper_params = hyper_params, 
                                                                    train_seed = seed, verbose = True,dict_to_bench=dict_to_bench) 

    return true_net,false_net,correlator_over_time_true,mean_test,mean_train,true_per,stocastic_per,


L2 = sys.argv[1]
L2 = int(L2)
global_num=100
edges_to_add=100
dict_to_bench={"device":'cuda','name':'scramble', "uniform":0,"L2_or_rand" :0,"stability_eigen":0,"device": "cuda"}

batch=global_num

true_net_list,false_net_list,correlator_over_time_true_list,mean_test_list,mean_train_list,true_per_list,stocastic_per_list=[],[],[],[],[],[],[]
for i in tqdm.tqdm(range(0,50)):
    dict_to_bench["num_epochs"]=5000
    dict_to_bench["batch"]=batch
    #dict_to_bench["l2"]=l2
    dict_to_bench["learning_rate"]=2.0e-3
    dict_to_bench["lr_final"]=1e-4
    dict_to_bench["seed"]=i
    dict_to_bench["edges_to_add"]=edges_to_add
    dict_to_bench["name"]=str(i)
    dict_to_bench["L2_norm"]=L2
    dict_to_bench["folder"]="parameter_study_mac_"+str(dict_to_bench["L2_norm"])+"xe-6"#macro,syn,ligands
    true_net,false_net,correlator_over_time_true,mean_test,mean_train,true_per,stocastic_per,=benchmark_this_dict(dict_to_bench)
    
    true_net_list.append(true_net)
    false_net_list.append(false_net)
    correlator_over_time_true_list.append(correlator_over_time_true)
    mean_test_list.append(mean_test)
    mean_train_list.append(mean_train)
    true_per_list.append(true_per)
    stocastic_per_list.append(stocastic_per)    

# Create the pandas DataFrame 
df_true_net = pd.DataFrame(true_net_list) 

df_false_net = pd.DataFrame(false_net_list) 

df_correlator_over_time_true = pd.DataFrame(correlator_over_time_true_list) 

df_mean_test = pd.DataFrame(mean_test_list) 

df_mean_train = pd.DataFrame(mean_train_list) 

true_per_df = pd.DataFrame(true_per_list) 

stocastic_per_df = pd.DataFrame(stocastic_per_list) 

#'LEMBAS/benchmark/figures/'


# Create the directory if it doesn't exist
output_dir = 'LEMBAS/benchmark/'+dict_to_bench["folder"]#macro,syn,ligands
os.makedirs(output_dir, exist_ok=True)

# Save as CSV files
df_true_net.to_csv(f'{output_dir}/true_net_with_macro_setup_self_prune.csv', index=False)
df_false_net.to_csv(f'{output_dir}/false_net_with_macro_setup_self_prune.csv', index=False)
df_correlator_over_time_true.to_csv(f'{output_dir}/correlator_over_time_true_with_macro_setup_self_prune.csv', index=False)
df_mean_test.to_csv(f'{output_dir}/mean_test_with_macro_setup_self_prune.csv', index=False)
df_mean_train.to_csv(f'{output_dir}/mean_train_with_macro_setup_self_prune.csv', index=False)
true_per_df.to_csv(f'{output_dir}/true_per_with_macro_setup_self_prune.csv', index=False)
stocastic_per_df.to_csv(f'{output_dir}/stocastic_per_with_macro_setup_self_prune.csv', index=False)

print("All DataFrames have been successfully saved.")

