import numpy
import torch
import matplotlib.pyplot as plt
import bionetwork
import plotting
import pandas
import saveSimulations
import time  # Add this import
import pickle
from scipy.stats import pearsonr
# --- Save data instead of plotting ---
import os
inputAmplitude = 3
projectionAmplitude = 1.2

#Setup optimizer
batchSize = 5
MoAFactor = 0.1
spectralFactor = 1e-3
maxIter = 5000
noiseLevel = 0
L2 = 1e-6



#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/macrophage-Model.tsv')
annotation = pandas.read_csv('data/macrophage-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
bionetParams = bionetwork.trainingParameters(iterations = 100, clipping=1, targetPrecision=1e-20, leak=0.01)
spectralCapacity = numpy.exp(numpy.log(1e-2)/bionetParams['iterations'])


inName = annotation.loc[annotation['ligand'],'code'].values
outName = annotation.loc[annotation['TF'],'code'].values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
outNameGene = [uniprot2gene[x] for x in outName]
nodeNameGene = [uniprot2gene[x] for x in nodeNames]

ligandInput = pandas.read_csv('data/macrophage-Ligands.tsv', sep='\t', low_memory=False, index_col=0)
TFOutput = pandas.read_csv('data/macrophage-TFs.tsv', sep='\t', low_memory=False, index_col=0)

#Subset input and output to intersecting nodes
inName = ligandInput.columns.values
outName = TFOutput.columns.values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
inNameGene = [uniprot2gene[x] for x in inName]
outNameGene = [uniprot2gene[x] for x in outName]
ligandInput = ligandInput.loc[:,inName]
TFOutput = TFOutput.loc[:,outName]

sampleName = ligandInput.index.values
model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
model.inputLayer.weights.requires_grad = False
model.network.preScaleWeights()


X = torch.tensor(ligandInput.values.copy(), dtype=torch.double)
Y = torch.tensor(TFOutput.values, dtype=torch.double)


#%%
criterion = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=0)
resetState = optimizer.state.copy()



stats = plotting.initProgressObject(maxIter)
N = X.shape[0]
curState = torch.rand((X.shape[0], model.network.bias.shape[0]), dtype=torch.double, requires_grad=False)
start_time = time.time()  # Start the timer


stats = plotting.finishProgress(stats)
# load model
model_state_dict = torch.load('model_for_eval/trainied_models/model_to_test_on_CPU_and_GPU_mac_state_dict.pth')


import scipy
# Ensure the assignment uses torch.nn.Parameter
model.inputLayer.weights = torch.nn.Parameter(model_state_dict['input_layer.weights'].cpu())



# Assume model_state_dict is already defined and contains the tensor
weights_tensor = model_state_dict['signaling_network.weights'].cpu()

# Extract all non-zero values from the 2D tensor and convert them to a LongTensor
non_zero_values = weights_tensor[weights_tensor != 0]

model.network.weights = torch.nn.Parameter(non_zero_values)
#plt.hist(non_zero_values, bins=20)
#plt.yscale('log')  # Set the y-axis to a log scale
#plt.show()



model.network.reset_A()


model.network.bias = torch.nn.Parameter(model_state_dict['signaling_network.bias'].cpu())
model.projectionLayer.weights = torch.nn.Parameter(model_state_dict['output_layer.weights'].cpu())

with open('model_for_eval/trainied_models/model_to_test_on_CPU_and_GPU_Y_hat.pickle', 'rb') as file:
    Y_gpu = pickle.load(file)


model.train()
Yhat,yfull = model(X)

loss_fn = torch.nn.MSELoss(reduction='mean')

fit_loss = loss_fn(Y, Yhat)# round the term first
print('loss')
print(fit_loss.item())
print("corr")
fit_loss.backward()
gradient_vector = model.network.weights.grad
gradient_bias = model.network.bias.grad#
with open('model_for_eval/trainied_models/model_to_test_on_CPU_and_GPU_gradient_vector.pickle', 'rb') as file:
    gradient_vector_gpu = pickle.load(file)

with open('model_for_eval/trainied_models/model_to_test_on_CPU_and_GPU_gradient_bias.pickle', 'rb') as file:
    gradient_bias_gpu = pickle.load(file)


# Flatten and detach all tensors for safe handling
grad_weight_cpu = gradient_vector.cpu().detach().flatten()
grad_weight_gpu = gradient_vector_gpu.cpu().detach().flatten()

grad_bias_cpu = gradient_bias.cpu().detach().flatten()
grad_bias_gpu = gradient_bias_gpu.cpu().detach().flatten()

Yhat_cpu = Yhat.cpu().detach().flatten()
Yhat_gpu = Y_gpu.cpu().detach().flatten()

# Compute correlations
corr_weights, _ = pearsonr(grad_weight_gpu, grad_weight_cpu)
corr_bias, _ = pearsonr(grad_bias_gpu, grad_bias_cpu)
corr_forward, _ = pearsonr(Yhat_gpu, Yhat_cpu)

print(f"Weights gradient correlation: {corr_weights}")
print(f"Bias gradient correlation: {corr_bias}")
print(f"Forward pass correlation: {corr_forward}")

# Create DataFrames
df_grad_weights = pandas.DataFrame({
    'gradient_gpu': grad_weight_gpu,
    'gradient_cpu': grad_weight_cpu
})
df_grad_bias = pandas.DataFrame({
    'gradient_gpu': grad_bias_gpu,
    'gradient_cpu': grad_bias_cpu
})
df_forward = pandas.DataFrame({
    'Y_gpu': Yhat_gpu,
    'Y_cpu': Yhat_cpu
})

# Add metadata
df_grad_weights.attrs['correlation'] = corr_weights
df_grad_bias.attrs['correlation'] = corr_bias
df_forward.attrs['correlation'] = corr_forward

# Save DataFrames to files
os.makedirs('model_for_eval/trainied_models/df_outputs', exist_ok=True)
df_grad_weights.to_csv('model_for_eval/trainied_models/df_outputs/gradient_weights.csv', index=False)
df_grad_bias.to_csv('model_for_eval/trainied_models/df_outputs/gradient_bias.csv', index=False)
df_forward.to_csv('model_for_eval/trainied_models/df_outputs/forward_pass.csv', index=False)

print("Saved DataFrames:")
print(" - gradient_weights.csv")
print(" - gradient_bias.csv")
print(" - forward_pass.csv")