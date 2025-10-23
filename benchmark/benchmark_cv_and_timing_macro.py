import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_bw, theme, geom_point, geom_hline, annotate, scale_y_log10
import plotnine as p9
import time  # Add this import at the top of your file
import tqdm
# Dynamically set the absolute path to the LEMBAS directory
sclembas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../LEMBAS'))
import sys
sys.path.insert(1, sclembas_path)
from LEMBAS.model.bionetwork import format_network, SignalingModel
from LEMBAS.benchmarking_version.benchmark_train import train_signaling_model
import LEMBAS.utilities as utils
from LEMBAS import plotting, io
# %% 
def benchmark_this_dict(dict_to_bench):
    # dict_to_bench is a dictionary that include name,uniform,L2_or_rand,stability_eigen
    # each of these variable control one of the test in benchmark_train. 
    # putting each variable to 0 will imply that the model will run as normal.
    # so 1 is to be read as the indicator for what to test
    # one extra variable is device. this variable control what device we will run our code on
    correlator,times=[],[]
    n_cores = 12
    utils.set_cores(n_cores)
    seed = 888# dict_to_bench['seed']
    if seed:
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        utils.set_seeds(seed = seed)
    
    device = dict_to_bench["device"]
    data_path = '../macrophage_data'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    keys_with_one = [key for key, value in dict_to_bench.items() if value == 1]
    # prior knowledge signaling network
    # Define data_path (adjust the relative path as necessary)
    # Define data_path (adjust the relative path as necessary)

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//macrophage_data'))
    # Load files using absolute paths
    net = pd.read_csv(os.path.join(data_path, 'macrophage-Model.tsv'), sep='\t', index_col=False)
    ligand_input = pd.read_csv(os.path.join(data_path, 'macrophage-Ligands.tsv'), sep='\t', low_memory=False, index_col=0)
    tf_output = pd.read_csv(os.path.join(data_path, 'macrophage-TFs.tsv'), sep='\t', low_memory=False, index_col=0)


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
    bionet_params = {'target_steps': 100, 'max_steps': 100, 'exp_factor':50, 'tolerance': 1e-5, 'leak':1e-2} # fed directly to model
    # training parameters
    lr_params = {'max_iter': 5000, 
                'learning_rate': 2e-3}
    other_params = {'batch_size': 10, 'noise_level': 10, 'gradient_noise_level': 1e-3}
    regularization_params = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-3, 'uniform_lambda_L2': 1e-4, 
                    'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-5}
    spectral_radius_params = {'n_probes_spectral': 2, 'power_steps_spectral': 5, 'subset_n_spectral': 2}
    target_spectral_radius = 0.9
    hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params} # fed into training function
    # Define the network
    mod = SignalingModel(net = net,
                        X_in = ligand_input,
                        y_out = tf_output, 
                        projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                        weight_label = weight_label, source_label = source_label, target_label = target_label,
                        bionet_params = bionet_params, 
                        dtype = torch.float32, device = device, seed = seed)
    # transform dataframe to tensors
    cut = dict_to_bench['cut']
    X_in = mod.df_to_tensor(mod.X_in)
    y_out = mod.df_to_tensor(mod.y_out)
    train_set=[ l for l in range(0,23) if l!=cut]
    test_set=[cut]
    X_test=X_in[test_set]
    y_test=y_out[test_set]
    mod.X_in=X_in[train_set]
    mod.y_out=y_out[train_set]
    # model setup
    mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
    mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius
    # loss and optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean') 
    optimizer = torch.optim.Adam
    start_time = time.time()  # Start timing
    mod, cur_loss, cur_eig, mean_loss, stats, X_train, X_t, X_val, y_train, y_t, y_val = train_signaling_model(mod, optimizer, loss_fn, 
                                                                    reset_epoch = 200, hyper_params = hyper_params, 
                                                                    train_seed = seed, verbose = True, dict_to_bench=dict_to_bench,train_split_frac = {'train': 1.0, 'test': 0.0, 'validation': None},) 
    end_time = time.time()  # End timing
    
    # store results
    data_path='benchmark/models'
    #data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//macrophage_data'))
    # Prepare model to be evaluated
    mod.eval()
    Y_hat, Y_full = mod(X_test)
# Assuming stats and other required variables are already defined
    # --- Figure 4: Prediction vs Actual on test set---
    y_pred = Y_hat.detach().flatten().cpu().numpy()
    y_actual = y_test.detach().flatten().cpu().numpy()
    pr, _ = pearsonr(y_pred, y_actual)
    print(f"Correlation score: {pr:.4f}")
    print(f"Training time: {end_time - start_time:.2f} seconds")  # Print the elapsed time
    timer=str(f'{end_time - start_time:.2f}')  # Convert timer to a float
    return timer,pr
# 
correlator=[]
times=[]
dict_to_bench={"name": "model_to_test_on_CPU_and_GPU",
                "uniform":0,
                "L2_or_rand" :0,
                "device": "cuda",
                "test_time":1,
                  'ProjectOutput_bias': 1,
                  'cut': 0,}
print('meta training started')
for i in tqdm.tqdm(range(0,23), desc='Benchmarking'):
    dict_to_bench['cut']=i
    timer,pr=benchmark_this_dict(dict_to_bench)
    
    times.append(timer)
    correlator.append(pr)
# Save into a DataFrame
df = pd.DataFrame({
    'cut': list(range(0, 23)),
    'time': times,
    'correlator': correlator
})

# Optional: save to CSV
df.to_csv("LEMBAS/data_from_LOOCV/macrophage_with_steady_and_reg.csv", index=False)

print("Benchmark results saved.")