

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv("LEMBAS/data_to_report/sup_fig_1_2_3/input_data_XX_YY_comparison_power_iter.csv")

x_data = df["x_data"].values
y_data = df["y_data"].values
plt.figure(figsize=(3, 3)) # Set the figure size for box plot
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
plt.savefig(f"LEMBAS/data_to_report/sup_fig_1_2_3/power_iter_correlator_box_plot.png", dpi=300)
