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
import tqdm
import seaborn as sns  # {{ edit_1 }}


# Dynamically set the absolute path to the LEMBAS directory
sclembas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../LEMBAS'))

import sys
sys.path.insert(1, sclembas_path)

from LEMBAS.benchmarking_version.benchmark_bionetwork_gradient_compare import format_network, SignalingModel
from LEMBAS.benchmarking_version.benchmark_train_gradient_compare import train_signaling_model
import LEMBAS.utilities as utils
from LEMBAS import plotting, io
# %% 


def benchmark_this_dict(dict_to_bench):
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
    bionet_params = {'target_steps': 100, 'max_steps': 150, 'exp_factor':21, 'tolerance': 1e-5, 'leak':1e-2} # fed directly to model


    other_params = {'batch_size': 8, 'noise_level': 10, 'gradient_noise_level': 1e-9}
    regularization_params = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                    'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-5}
    spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 5, 'subset_n_spectral': 10}
    target_spectral_radius = 0.8
    



    print("Training has started")
    print("-----------------------------------------------------")
    print("Printing training progress")
    print("-----------------------------------------------------")
    power_steps_spectral=[1]#,3,5,10]
    box_plot_correlator=[]
    max_iter_=[100]
    # training loop
    Grad1=[]
    Grad2=[]
    to_loop=[1]
    for i in tqdm.tqdm(range(0,len(power_steps_spectral))):
        print('run number '+ str(i))
            # training parameters
        lr_params = {'max_iter': max_iter_[0], 
                    'learning_rate': 2e-3}
        spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': power_steps_spectral[i], 'subset_n_spectral': 10}
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


        output_dir = "LEMBAS/data_to_report/figs"
        # model setup
        mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
        mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

        # loss and optimizer
        loss_fn = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam
        mod, mod_old_eigen,grad1,grad2,correlator= train_signaling_model(mod, optimizer, loss_fn, 
                                                                        reset_epoch = 200, hyper_params = hyper_params, 
                                                                        train_seed = seed, verbose = True,dict_to_bench=dict_to_bench) 
        box_plot_correlator.append(correlator)

        print("-----------------------------------------------------")
        print("Training done, printing started")
        print("-----------------------------------------------------")
        Grad1.append(grad1)
        Grad2.append(grad2)
        print('test')


   # Box plot for box_plot_correlator
    if dict_to_bench['test_eigen_1_or_uni_0']==1:
        plt.figure(figsize=(3, 3)) # Set the figure size for box plot

        # Create a list of lists based on power_steps_spectral
        list_of_lists = [[(power_steps_spectral[i])] * len(box_plot_correlator[i]) for i in range(len(power_steps_spectral))]

        # Flatten the lists for boxplot
        x_data = np.concatenate(list_of_lists)
        y_data = np.concatenate(box_plot_correlator)
        # Save the x_data and y_data
        np.save(f"{output_dir}/{dict_to_bench['name']}_x_data.npy", x_data)
        np.save(f"{output_dir}/{dict_to_bench['name']}_y_data.npy", y_data)
        # Create box plot with seaborn
        # Create box plot with seaborn, adding transparency
        sns.boxplot(x=x_data, y=y_data, showfliers=False, color='white', saturation=0, linewidth=1,
                boxprops=dict(facecolor='white', alpha=0.3),   # Box transparency
                whiskerprops=dict(alpha=0.5),   # Whiskers transparency
                capprops=dict(alpha=0.5),       # Caps transparency
                medianprops=dict(color='red', linewidth=2, alpha=0.7))  # Median line transparency


        # Get the unique x-values from the boxplot for alignment
        unique_x_values = np.arange(len(np.unique(x_data)))

        # Overlay scatter plot
        # Map the original categorical values to the corresponding numerical positions used by the boxplot
        x_positions = [unique_x_values[np.where(np.unique(x_data) == x)[0][0]] for x in x_data]

        plt.scatter(x_positions, y_data, color='blue', alpha=0.2)  # Scatter plot overlaid

        # Set the x-ticks to the original labels (power_steps_spectral)
        plt.xticks(unique_x_values, np.unique(x_data))

        plt.title('Box Plot of Correlator Values')  # Title of the box plot
        plt.xlabel('Iteration steps in Power iterations')  # X-axis label
        plt.ylabel('Correlation')  # Y-axis label
        plt.grid(True)  # Add grid for better readability

        # Save the box plot
        plt.tight_layout()
        plt.savefig(f"LEMBAS/data_to_report/figs/{dict_to_bench['name']}_correlator_box_plot.png", dpi=300)
        plt.clf()
        plt.plot(grad1,grad2)
        plt.show()
        print('saved PI')
    else:
        print("Figures are saved")
        print("-----------------------------------------------------")

        

        # Calculate slope k with zero intercept
        grad1 = np.array(grad1)
        grad2 = np.array(grad2)
        df = pd.DataFrame({'grad1': grad1, 'grad2': grad2})
        df.to_csv("LEMBAS/data_to_report/sup_fig_1_2_3/input_data_XX_YY_comparison_uniform.csv", index=False)



dict_to_bench={"name": "uniform", "L2_or_rand" :0,"test_eigen_1_or_uni_0":0,"device": "cuda"}
                          
benchmark_this_dict(dict_to_bench)

