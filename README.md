# LEMBAS

This is a re-implementation of LEMBAS. See the [github](https://github.com/Lauffenburger-Lab/LEMBAS) and associated [manuscript](https://doi.org/10.1038/s41467-022-30684-y) for details. See the [Documentation](https://hmbaghdassarian.github.io/LEMBAS/) and [tutorial](https://hmbaghdassarian.github.io/LEMBAS/macrophage_example/) for implementation.

## Installation instructions

Pull LEMBAS from [PyPI](https://pypi.org/project/LEMBAS-re/) (consider using pip3 to access Python 3):

```
pip install LEMBAS-re
```

# Tutorial 


To get started with **LEMBAS**, we recommend running the following notebook:

To **modify LEMBAS and its functions**, use the versions located in:

LEMBAS/LEMBAS/model

The other folders contain of function and scripts to test self pruning and compare our GPU implementation to the original CPU version.

# LEMBAS Figure Regeneration Guide

## Conventions and download of data
### Naming Convention

To maintain consistency with **LEMBAS CPU**, the following naming conventions are used:

| Name in Paper          | Name in Code        |
|-------------------------|---------------------|
| High coverage dataset   | Ligand dataset      |
| Low coverage dataset    | Macrophage dataset  |
| Synthetic dataset       | Synthetic dataset   |


### Download all the data for figures 
#### Zenodo
If you are not interested in regenerating all the models i.e 450 models used to prove self-pruning or other results that take longer than 1 minute to generate please download these using Zenodo: https://zenodo.org/records/17425598

| Name of folder in Zenodo          | Folder to put data in       |
|-------------------------|---------------------|
| Cross_Validation   | LEMBAS/data_to_report/data_and_anlysis_from_LOOCV      |
| parameter_study_between_conditions    | LEMBAS/data_to_report/fig_1/model_compare  |
| trained models       | LEMBAS/data_to_report/fig_2_3   |
| sup_fig_1_2_3       | LEMBAS/data_to_report/sup_fig_1_2_3   |


#### Data to train models on 
Data to train the models can be found at: [https://github.com/Lauffenburger-Lab/LEMBAS]

## Generate data for figures yourself
To regenerate the data used in the paper follow these steps. 

#### Figure 1's data 
Run these scripts to generate the data

LEMBAS/benchmark/benchmark_cv_and_timing_ligands.py

LEMBAS/benchmark/self_pruning_figure_macrophage_and_syn_setup.py

LEMBAS/benchmark/benchmarking_bias.ipynb

LEMBAS/benchmark/benchmarking_models.ipynb

#### Figure 2's data
For this figure you have define what datasets interest you and what L2 norms. The Ligand dataset is treated separately from the macrophage and synthetic data sets as hyperparameters differs. 

LEMBAS/benchmark/self_pruning_figure_ligand_setup.py

LEMBAS/benchmark/self_pruning_figure_macrophage_and_syn_setup.py

## Create figures

#### Generate figures for paper

Figure 1 LEMBAS/data_to_report/fig_1 and LEMBAS/data_to_report/model_compare

Figure 2 LEMBAS/data_to_report/fig_2_3

#### Generate supplementary figures
For Sup Figure 1, you need to move the contents of LEMBAS/data_to_report/sup_fig_1_2_3/put_in_CPU_if_replication_of_back_forward/model_for_eval/trainied_models into the original LEMBAS model repository. This is because we benchmark against a real model.

To train that model, run LEMBAS/benchmark/benchmark_compare_cpu_and_gpu_gradient.py, and then move the relevant model folder.

If you are uninterested in doing this process, we refer to Zenodo.

To plot the results, use LEMBAS/data_to_report/sup_fig_1_2_3/put_in_CPU_if_replication_of_back_forward/sup_fig_1.py.

The orginal implementation can be found at https://github.com/Lauffenburger-Lab/LEMBAS

Sup figure 2 and 3 run LEMBAS/benchmark/sup_fig_2_data.py and LEMBAS/benchmark/sup_fig_3_data.py and than LEMBAS/data_to_report/sup_fig_1_2_3/sup_fig_2.py LEMBAS/data_to_report/sup_fig_1_2_3/sup_fig_3.py

The rest of the sup figure use the same data as the orginal figures. 

Sup figure 4 and 5 LEMBAS/data_to_report/fig_1/sup_fig_4_5.py

Sup figure 6 and 7 LEMBAS/data_to_report/fig_1/model_compare/fig_sup_6_7.py

Sup figure 8 LEMBAS/data_to_report/fig_2_3/fig_2_c_sup_fig_8.py.py

Sup figure 9 LEMBAS/data_to_report/fig_2_3/fig_2_d.py

Sup figure X and Y LEMBAS/data_to_report/fig_2_3/sup_fig_X_Y.py.py
