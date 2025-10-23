import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("LEMBAS/data_to_report/sup_fig_1_2_3/input_data_XX_YY_comparison_uniform.csv")

grad1 = df["grad1"].values
grad2 = df["grad2"].values
k = np.sum(grad1 * grad2) / np.sum(grad1 ** 2)

# Compute Pearson correlation
corr = np.corrcoef(grad1, grad2)[0, 1]
plt.figure(figsize=(3, 3))  # Create a single figure
# Scatter plot
plt.scatter(grad1, grad2, color='blue', alpha=0.2)

# Plot trend line y = k * x
x_vals = np.array([grad1.min(), grad1.max()])
y_vals = k * x_vals




# Plot trend line
plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'y = {k:.3f} x')

plt.title('State regularization ')
plt.xlabel('new regularization')
plt.ylabel('old regularization')
plt.grid(True)
plt.legend()

# Annotate correlation
plt.text(0.05, 0.95, f'corr = {corr:.4f}', transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))

plt.tight_layout()
plt.show()
plt.savefig(f"LEMBAS/data_to_report/sup_fig_1_2_3/unif_gradient_comparison.png", dpi=300)

print('uniform gradient comparison saved')
plt.clf()