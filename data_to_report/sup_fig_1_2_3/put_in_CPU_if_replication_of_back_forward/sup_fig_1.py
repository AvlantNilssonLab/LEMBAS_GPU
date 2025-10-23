import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import os

# Load DataFrames
base_path = 'LEMBAS/data_to_report/sup_fig_1_2_3/put_in_CPU_if_replication_of_back_forward/model_for_eval/trainied_models/df_outputs'

df_grad_weights = pd.read_csv(os.path.join(base_path, 'gradient_weights.csv'))
df_grad_bias = pd.read_csv(os.path.join(base_path, 'gradient_bias.csv'))
df_forward = pd.read_csv(os.path.join(base_path, 'forward_pass.csv'))

# --- Compute correlations ---
corr_weights, _ = pearsonr(df_grad_weights['gradient_gpu'], df_grad_weights['gradient_cpu'])
corr_bias, _ = pearsonr(df_grad_bias['gradient_gpu'], df_grad_bias['gradient_cpu'])
corr_forward, _ = pearsonr(df_forward['Y_gpu'], df_forward['Y_cpu'])

print(f"Weights gradient correlation: {corr_weights}")
print(f"Bias gradient correlation: {corr_bias}")
print(f"Forward pass correlation: {corr_forward}")

# --- Plot ---
plt.figure(figsize=(15, 5))

# Plot 1: Gradients of weights
plt.subplot(1, 3, 1)
x = df_grad_weights['gradient_gpu']
y = df_grad_weights['gradient_cpu']
plt.plot(x, y, '.', label='Data Points')
coeffs = np.polyfit(x, y, 1)
trend = np.poly1d(coeffs)
plt.plot(x, trend(x), 'r-', label=f'Trendline (slope={coeffs[0]:.2f})')
plt.xlabel('Gradient (GPU)')
plt.ylabel('Gradient (CPU)')
plt.title(f'Gradients of weights\nCorr = {corr_weights:.4f}')
plt.legend()

# Plot 2: Gradients of bias
plt.subplot(1, 3, 2)
x = df_grad_bias['gradient_gpu']
y = df_grad_bias['gradient_cpu']
plt.plot(x, y, '.', label='Data Points')
coeffs = np.polyfit(x, y, 1)
trend = np.poly1d(coeffs)
plt.plot(x, trend(x), 'r-', label=f'Trendline (slope={coeffs[0]:.2f})')
plt.xlabel('Gradient (GPU)')
plt.ylabel('Gradient (CPU)')
plt.title(f'Gradients of bias\nCorr = {corr_bias:.4f}')
plt.legend()

# Plot 3: Forward pass
plt.subplot(1, 3, 3)
x = df_forward['Y_gpu']
y = df_forward['Y_cpu']
plt.plot(x, y, '.', label='Data Points')
coeffs = np.polyfit(x, y, 1)
trend = np.poly1d(coeffs)
plt.plot(x, trend(x), 'r-', label=f'Trendline (slope={coeffs[0]:.2f})')
plt.xlabel('Y (GPU)')
plt.ylabel('Y (CPU)')
plt.title(f'Forward pass\nCorr = {corr_forward:.4f}')
plt.legend()

plt.tight_layout()
plt.savefig("LEMBAS/data_to_report/sup_fig_1_2_3/cpu_gpu_correlation_forward_backward.png", dpi=300, bbox_inches='tight')