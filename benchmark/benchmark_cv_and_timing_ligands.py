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
from LEMBAS.benchmarking_version.benchmark_train_ligand import train_signaling_model
import LEMBAS.utilities as utils
from LEMBAS import plotting, io
import multiprocessing as mp
import numpy as np
import pandas as pd
import tqdm
import os
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
    data_path = '../ligand_data'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    keys_with_one = [key for key, value in dict_to_bench.items() if value == 1]
    # prior knowledge signaling network
    # Define data_path (adjust the relative path as necessary)
    # Define data_path (adjust the relative path as necessary)

                # Define data_path (adjust the relative path as necessary)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//ligand_data'))

    # Load files using absolute paths
    net = pd.read_csv(os.path.join(data_path, 'ligandScreen-Model.tsv'), sep='\t', index_col=False)
    ligand_input = pd.read_csv(os.path.join(data_path, 'ligandScreen-Ligands.tsv'), sep='\t', low_memory=False, index_col=0)
    tf_output = pd.read_csv(os.path.join(data_path, 'ligandScreen-TFs.tsv'), sep='\t', low_memory=False, index_col=0)



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
    bionet_params = {'target_steps': 150, 'max_steps': 150, 'exp_factor':10, 'tolerance': 1e-5, 'leak':1e-2} # fed directly to model
    # training parameters
        
    lr_params = {'max_iter': 25000, 
                'learning_rate': 1e-3}
    other_params = {'batch_size': 150, 'noise_level': 1.0, 'gradient_noise_level': 0.001}
    regularization_params = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-4, 'uniform_lambda_L2': 1e-5, 
                    'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-3}
    spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 5, 'subset_n_spectral': 2}
    target_spectral_radius = 0.9
    """    lr_params = {'max_iter': 8000, 
                'learning_rate': 1e-3}
    other_params = {'batch_size': 150, 'noise_level': 10.0, 'gradient_noise_level': 0.1}
    regularization_params = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-4, 'uniform_lambda_L2': 1e-5, 
                    'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-3}
    spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 5, 'subset_n_spectral': 2}
    target_spectral_radius = 0.9
    """
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

    X_in = mod.df_to_tensor(mod.X_in)
    y_out = mod.df_to_tensor(mod.y_out)
    # load the test and train based on index from dict_to_bench['cut']
    # Load the TSV file that contains the index and condition
    df = pd.read_csv(os.path.join(data_path, 'conditions.tsv'), sep='\t', index_col=False)


    # Loop over each fold for cross-validation

    # Extract the test set names
    test_set = df[df['Index'] == dict_to_bench['cut']]['Condition'].to_list()

    # Find indices of test_set elements in ligand_input
    # Find the indices in ligand_inputs that match any element in test_set
    test_row_names = ligand_input.index[ligand_input.index.isin(test_set)].tolist()

    test_indices = ligand_input.index.get_indexer(test_row_names)
    # Find the train indices (all other indices)
    train_row_names = ligand_input.index.difference(test_row_names).tolist()
    train_indices = ligand_input.index.get_indexer(train_row_names)

    
    X_train = (X_in[train_indices])
    y_train = (y_out[train_indices])
    X_test = (X_in[test_indices])
    y_test = (y_out[test_indices])

    # Set the training and testing data in your model
    mod.X_in = X_train
    mod.y_out = y_train
    # model setup
    mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
    mod.signaling_network.prescale_weights(target_radius = 0.7) # spectral radius
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
    Y_hat, _ = mod.deprecation_forward_no_bias(X_test)
    Y_hat_train, _ = mod.deprecation_forward_no_bias(X_train)  # Assuming you have X_train for training set evaluation

    # --- Figure 4: Prediction vs Actual on test set---
    y_pred = Y_hat.detach().flatten().cpu().numpy()
    y_actual = y_test.detach().flatten().cpu().numpy()
    pr_test, _ = pearsonr(y_pred, y_actual)

    # --- Compute correlation for training set ---
    y_pred_train = Y_hat_train.detach().flatten().cpu().numpy()
    y_actual_train = y_train.detach().flatten().cpu().numpy()
    pr_train, _ = pearsonr(y_pred_train, y_actual_train)

    print(pr_test)
    print(pr_train)
    print(f"Training time: {end_time - start_time:.2f} seconds")  # Print the elapsed time
    timer = str(f'{end_time - start_time:.2f}')  # Convert timer to a string

    return timer, pr_test, pr_train
# 
correlator_test = []
correlator_train = []
times = []




# Function to run benchmarking for a subset of indices
def run_benchmark_subset(start_idx, end_idx, return_dict):
    local_times = []
    local_correlator_test = []
    local_correlator_train = []
    local_cuts = []  # New list to store cut values
   
    for i in tqdm.tqdm(range(start_idx, end_idx), desc=f'Benchmarking {start_idx}-{end_idx-1}'):
        dict_to_bench = {
            "name": "model_to_test_on_CPU_and_GPU",
            "uniform": 0,
            "L2_or_rand": 0,
            "device": "cuda",
            "test_time": 1,
            'ProjectOutput_bias': 0,
            'cut': 0,
        }
        dict_to_bench['cut'] = i
       
        timer, pr_test, pr_train = benchmark_this_dict(dict_to_bench)
       
        local_times.append(timer)
        local_correlator_test.append(pr_test)
        local_correlator_train.append(pr_train)
        local_cuts.append(i)  # Store the cut value
       
        # Save intermediate results to avoid data loss in case of crashes
        df = pd.DataFrame({
            "times": local_times,
            "correlator_test": local_correlator_test,
            "correlator_train": local_correlator_train,
            "indices": list(range(start_idx, start_idx + len(local_times))),
            "cuts": local_cuts  # Add cuts to the intermediate CSV
        })
        df.to_csv(f"LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/new_Ligand_benching_test_{start_idx}_to_{i}_new_RE_200_E_25000.csv", index=False)
   
    return_dict[start_idx] = {
        'times': local_times,
        'correlator_test': local_correlator_test,
        'correlator_train': local_correlator_train,
        'indices': list(range(start_idx, end_idx)),
        'cuts': local_cuts  # Add cuts to the returned dictionary
    }

if __name__ == "__main__":
    # Create directory if it doesn't exist
    os.makedirs("LEMBAS/data_from_LOOCV", exist_ok=True)
    print("New Ligand Trainer")
    # Total range to process
    total_range = 28
   
    # Split the work into 7 processes
    chunks = 1
    chunk_size = total_range // chunks
    remainder = total_range % chunks
   
    # Calculate start and end indices for each process
    ranges = []
    start = 0
    for i in range(chunks):
        # Add one extra item to earlier chunks if division isn't even
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        ranges.append((start, end))
        start = end
   
    # Use Manager to share results between processes
    manager = mp.Manager()
    return_dict = manager.dict()
   
    # Create and start processes
    processes = []
    for start_idx, end_idx in ranges:
        p = mp.Process(target=run_benchmark_subset, args=(start_idx, end_idx, return_dict))
        processes.append(p)
        p.start()
   
    # Wait for all processes to complete
    for p in processes:
        p.join()
   
    # Combine results
    all_times = []
    all_correlator_test = []
    all_correlator_train = []
    all_indices = []
    all_cuts = []  # New list for cuts
   
    for start_idx in sorted(return_dict.keys()):
        results = return_dict[start_idx]
        all_times.extend(results['times'])
        all_correlator_test.extend(results['correlator_test'])
        all_correlator_train.extend(results['correlator_train'])
        all_indices.extend(results['indices'])
        all_cuts.extend(results['cuts'])  # Add cuts from each process
   
    # Sort results based on indices
    sorted_data = sorted(zip(all_indices, all_times, all_correlator_test, all_correlator_train, all_cuts))
    sorted_indices, sorted_times, sorted_test, sorted_train, sorted_cuts = zip(*sorted_data)
   
    # Create final DataFrame
    final_df = pd.DataFrame({
        "times": sorted_times,
        "correlator_test": sorted_test,
        "correlator_train": sorted_train,
        "cuts": sorted_cuts  # Include cuts in the final CSV
    })
   
    # Save final results
    #final_df.to_csv("LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/new_Ligand_benching_test_combined_same_as_old_new_split_25000_150_with_noise_later_peak.csv", index=False)
    final_df.to_csv("LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/ligand_with_new_setup.csv", index=False)
   
    print("All benchmarking complete and results combined!")