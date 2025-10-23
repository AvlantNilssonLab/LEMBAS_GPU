import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import fligner
# Assuming cohen_d is defined
def cohens_d(x, y):
    """Compute Cohen's d for independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std
# List of your CSV file paths
file_paths = [
    #'LEMBAS/data_and_anlysis_from_LOOCV/macrophage_without_steady_and_reg.csv', # GPU without PI
    'LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/macrophage_CPU.csv'  ,                   # CPU
    'LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/macrophage_with_steady_and_reg.csv',   # GPU with PI

]

labels = ['CPU','GPU']#'GPU w/o PI and State',

# Read data
data = [pd.read_csv(path) for path in file_paths]

# Extract correlators and times
correlators = [df['correlator'].values for df in data]
times = [df['time'].values for df in data]
# Annotate medians


# Boxplot for correlators with individual samples and connecting lines
fig, ax = plt.subplots(figsize=(3, 2))  # fixed figure size
# Transpose data to get per-sample data across conditions
correlator_array = np.array(correlators)
if correlator_array.shape[1] != 3:  # Transpose if necessary
    correlator_array = correlator_array.T


# Assigning data for clarity
cpu_corr,gpu_PI_corr = correlators
cpu_sec, gpu_PI_sec = times

# Comparisons
comparisons = [
    #("CPU", "GPU w/o PI", gpu_no_PI_corr, cpu_corr, "Correlation"),
    ("CPU", "GPU w/ PI", cpu_corr, gpu_PI_corr, "Correlation"),
    #("GPU w/o PI", "GPU w/ PI", gpu_no_PI_corr, gpu_PI_corr, "Correlation"),

    #("CPU", "GPU w/o PI", gpu_no_PI_sec, cpu_sec, "Execution Time"),
    ("CPU", "GPU w/ PI", cpu_sec, gpu_PI_sec, "Execution Time"),
    #("GPU w/o PI", "GPU w/ PI", gpu_no_PI_sec, gpu_PI_sec, "Execution Time"),
]

# Run Wilcoxon and Cohen's d
wilcoxon_results = []
cohen_list=[]
for name1, name2, data1, data2, metric in comparisons:
    stat, p_value = wilcoxon(data1, data2, alternative='two-sided')
    d = cohens_d(data1, data2)
    cohen_list.append(d)
    sig = "Significant" if p_value < 0.05 else "Not Significant"
    wilcoxon_results.append([f"{name1} vs {name2}", metric, f"{p_value:.4f}", f"{d:.2f}", sig])


df_wilcoxon = pd.DataFrame(wilcoxon_results, columns=["Comparison", "Metric", "p-value", "Cohen's d", "Significance"])
df_wilcoxon.to_csv("wilcoxon_results.csv", index=False)

# Scatter individual points
for i in range(2):
    y = correlator_array[:, i]
    x = np.random.normal(i + 1, 0.01, size=len(y))  # Jitter x-axis
    plt.scatter(x, y, alpha=0.7, label=labels[i])

# Add medians
for i, data_group in enumerate(correlators):
    median = np.median(data_group)
    plt.text(i + 1 + 0.25, median-median*0.1, f'{median:.2f}', ha='center', va='bottom', fontsize=10, color='black')

plt.boxplot(correlators, labels=labels, showfliers=True, medianprops=dict(color='black'))

# Wilcoxon p-value
corr_p_value = float(df_wilcoxon.loc[df_wilcoxon['Metric'] == 'Correlation', 'p-value'].values[0])
plt.text(1.5, -0.5, f'Wilcoxon p = {corr_p_value:.4f}', color='black', ha='center', fontsize=11)

# Fligner–Killeen p-value (variance equality)
try:
    # Assuming correlators is a list of arrays/lists, one per group
    plt.text(1.5, -0.8, f"Cohen's d = {cohen_list[0]:.2f}", color='black', ha='center', fontsize=11)
    
except ValueError as e:
    print("Fligner–Killeen test failed:", e)

plt.ylabel('Correlation')
plt.ylim(-1, 1.2)
fig.subplots_adjust(
    left=0.15,   # space from figure left edge to axes
    right=0.95,  # space from figure right edge to axes
    top=0.90,    # space from figure top edge to axes
    bottom=0.15  # space from figure bottom edge to axes
)

plt.savefig("LEMBAS/data_to_report/figs/correlators_boxplot.svg", dpi=300, bbox_inches='tight')
plt.close()

# Boxplot for execution times with individual samples and connecting lines
fig, ax = plt.subplots(figsize=(3, 2))  # fixed figure size
time_array = np.array(times)
if time_array.shape[1] != 3:  # Transpose if needed
    time_array = time_array.T


# Scatter individual points
for i in range(2):
    y = time_array[:, i]
    x = np.random.normal(i + 1, 0.01, size=len(y))  # Jitter x-axis
    plt.scatter(x, y, alpha=0.7, label=labels[i])

# Add medians
for i, data_group in enumerate(times):
    median = np.median(data_group)
    plt.text(i + 1 + 0.25, median-median*0.1, f'{median:.0f}', ha='center', va='bottom', fontsize=10, color='black')

plt.boxplot(times, labels=labels, showfliers=True, medianprops=dict(color='black'))
# Annotate Wilcoxon p-value on execution time plot
time_p_value = float(df_wilcoxon.loc[df_wilcoxon['Metric'] == 'Execution Time', 'p-value'].values[0])
plt.text(1.5, 300, f'Wilcoxon p = {time_p_value:.4f}', color='black', ha='center', fontsize=11)
cpu_mean = np.mean(cpu_sec)
gpu_mean = np.mean(gpu_PI_sec)
print("CPU mean time:", cpu_mean, "GPU mean time:", gpu_mean)
speed_up = cpu_mean / gpu_mean if gpu_mean != 0 else np.nan
ax.text(1.5, 100, f'Speed-up = {speed_up:.2f}x', color='black', ha='center', fontsize=11)
plt.ylabel('Time (s)')
plt.ylim(0, 1600)
#plt.grid(True)
fig.subplots_adjust(
    left=0.15,   # space from figure left edge to axes
    right=0.95,  # space from figure right edge to axes
    top=0.90,    # space from figure top edge to axes
    bottom=0.15  # space from figure bottom edge to axes
)

plt.savefig("LEMBAS/data_to_report/figs/Time_boxplot.svg", dpi=300, bbox_inches='tight')
plt.close()




# Summary statistics
summary_stats = {
    "Method": [ "GPU","CPU"],
    "Mean Seconds": [ np.mean(gpu_PI_sec),np.mean(cpu_sec)],
    "Mean Correlation": [np.mean(gpu_PI_corr),np.mean(cpu_corr)],
}
df_summary = pd.DataFrame(summary_stats)

# Create the figure with 2 subplots for tables
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
ax1.axis('off')
ax2.axis('off')
# Format the values in df_summary to show up to 5 characters (e.g., 2 decimal places)
df_summary_formatted = df_summary.copy()
df_summary_formatted["Mean Seconds"] = df_summary_formatted["Mean Seconds"].map(lambda x: f"{x:.2f}")
df_summary_formatted["Mean Correlation"] = df_summary_formatted["Mean Correlation"].map(lambda x: f"{x:.4f}")

# First table: Summary statistics (with formatted values)
table1 = ax1.table(cellText=df_summary_formatted.values,
                   colLabels=df_summary_formatted.columns,
                   cellLoc='center',
                   loc='center')
table1.auto_set_font_size(False)
table1.set_fontsize(12)
table1.scale(1, 1.5)
ax1.set_title("Summary Statistics", fontweight='bold')

# Second table: Wilcoxon and Cohen's d
table2 = ax2.table(cellText=df_wilcoxon.values,
                   colLabels=df_wilcoxon.columns,
                   cellLoc='center',
                   loc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 1.8)
ax2.set_title("Wilcoxon Test & Effect Size", fontweight='bold')

# Save or show
fig.subplots_adjust(
    left=0.15,   # space from figure left edge to axes
    right=0.95,  # space from figure right edge to axes
    top=0.90,    # space from figure top edge to axes
    bottom=0.15  # space from figure bottom edge to axes
)

plt.savefig("LEMBAS/data_to_report/figs/table_1and2.png", dpi=300, bbox_inches='tight')
plt.close()

from scipy.stats import ttest_rel  # For paired data

# Add t-test to existing comparisons
ttest_results = []
for name1, name2, data1, data2, metric in comparisons:
    t_stat, t_p = ttest_rel(data1, data2)
    sig = "Significant" if t_p < 0.05 else "Not Significant"
    ttest_results.append([f"{name2} vs {name1}", metric, f"{t_p:.4f}", sig])

# Convert to DataFrame
df_ttest = pd.DataFrame(ttest_results, columns=["Comparison", "Metric", "t-test p-value", "Significance"])

# Create table for t-test results
fig, ax = plt.subplots(figsize=(6, 5))
ax.axis('off')

table3 = ax.table(cellText=df_ttest.values,
                  colLabels=df_ttest.columns,
                  cellLoc='center',
                  loc='center')
table3.auto_set_font_size(False)
table3.set_fontsize(10)
table3.scale(1, 1.5)
ax.set_title("Paired t-test Results", fontweight='bold')

fig.subplots_adjust(
    left=0.15,   # space from figure left edge to axes
    right=0.95,  # space from figure right edge to axes
    top=0.90,    # space from figure top edge to axes
    bottom=0.15  # space from figure bottom edge to axes
)

plt.savefig("LEMBAS/data_to_report/figs/table_3.png", dpi=300, bbox_inches='tight')
plt.close()

# Optionally, save results
df_ttest.to_csv("ttest_results.csv", index=False)