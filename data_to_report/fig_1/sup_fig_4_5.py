import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import itertools
# Assuming cohen_d is defined
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)



def add_pval_annotations(ax, data_groups, comparisons, df_results, y_offset=0.04, metric="Correlation"):
    label_map = {
        "GPU w/o Reg": "GPU w/o PI",
        "GPU": "GPU w/ PI",
        "CPU": "CPU"
    }

    for i, (g1, g2, idx1, idx2) in enumerate(comparisons):
        g1_mapped = label_map.get(g1, g1)
        g2_mapped = label_map.get(g2, g2)

        row = df_results[(df_results["Comparison"] == f"{g1_mapped} vs {g2_mapped}") &
                         (df_results["Metric"] == metric)]
        if row.empty:
            row = df_results[(df_results["Comparison"] == f"{g2_mapped} vs {g1_mapped}") &
                             (df_results["Metric"] == metric)]

        if row.empty:
            print(f"⚠️ No match for {g1} vs {g2} in {metric}")
            continue

        pval = float(row["p-value"].values[0])

        # Below axis position (in axis coordinates)
        y = -(i + 1) * y_offset
        x1, x2 = idx1 + 1, idx2 + 1
        mid = (x1 + x2) / 2
        print(y)
        # Bracket below x-axis
        ax.plot([x1, x1, x2, x2],
                [y-0.025+0.45, y - y_offset/3+0.45, y - y_offset/3+0.45, y-0.025+0.45],
                lw=1.2, c='k',
                transform=ax.get_xaxis_transform(),
                clip_on=False)

        # P-value text below bracket
        ax.text(mid, y - y_offset/2+0.45, f"p = {pval:.3f}",
                ha='center', va='top', fontsize=9, color="black",
                transform=ax.get_xaxis_transform(),
                clip_on=False)

    # Expand figure bottom margin (otherwise annotations may be cut off)
    ax.figure.subplots_adjust(bottom=0.25)


# List of your CSV file paths
file_paths = [
    'LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/macrophage_without_steady_and_reg.csv', # GPU without PI
    'LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/macrophage_with_steady_and_reg.csv',   # GPU with PI
    'LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/macrophage_CPU.csv'                     # CPU
]

labels = ["GPU w/o Reg",'GPU', 'CPU']#'GPU w/o PI and State',

# Read data
data = [pd.read_csv(path) for path in file_paths]

# Extract correlators and times
correlators = [df['correlator'].values for df in data]
times = [df['time'].values for df in data]
# Annotate medians

# Boxplot for correlators with individual samples and connecting lines
plt.figure(figsize=(3, 3))
# Transpose data to get per-sample data across conditions
correlator_array = np.array(correlators)
if correlator_array.shape[1] != 3:  # Transpose if necessary
    correlator_array = correlator_array.T


# Assigning data for clarity
gpu_no_PI_corr,gpu_PI_corr, cpu_corr = correlators
gpu_no_PI_sec,gpu_PI_sec, cpu_sec = times


# Comparisons
comparisons = [
    ("CPU", "GPU w/o PI", gpu_no_PI_corr, cpu_corr, "Correlation"),
    ("CPU", "GPU w/ PI", cpu_corr, gpu_PI_corr, "Correlation"),
    ("GPU w/o PI", "GPU w/ PI", gpu_no_PI_corr, gpu_PI_corr, "Correlation"),

    ("CPU", "GPU w/o PI", gpu_no_PI_sec, cpu_sec, "Execution Time"),
    ("CPU", "GPU w/ PI", cpu_sec, gpu_PI_sec, "Execution Time"),
    ("GPU w/o PI", "GPU w/ PI", gpu_no_PI_sec, gpu_PI_sec, "Execution Time"),
]

# Run Wilcoxon and Cohen's d
wilcoxon_results = []
for name1, name2, data1, data2, metric in comparisons:
    stat, p_value = wilcoxon(data1, data2, alternative='two-sided')
    d = cohen_d(data1, data2)
    sig = "Significant" if p_value < 0.05 else "Not Significant"
    wilcoxon_results.append([f"{name1} vs {name2}", metric, f"{p_value:.4f}", f"{d:.2f}", sig])

df_wilcoxon = pd.DataFrame(wilcoxon_results, columns=["Comparison", "Metric", "p-value", "Cohen's d", "Significance"])
df_wilcoxon.to_csv("wilcoxon_results.csv", index=False)
color_list=["g","tab:blue","tab:orange"]

# Example for correlator plot
plt.figure(figsize=(3, 3))
for i in range(3):
    y = correlators[i]
    x = np.random.normal(i + 1, 0, size=len(y))
    plt.scatter(x, y, alpha=0.7, label=labels[i], color=color_list[i])

# Make boxplot and get its stats
box = plt.boxplot(correlators, labels=labels, showfliers=False,
                  medianprops=dict(color='black'))

# Add medians as text
medians = [np.median(vals) for vals in correlators]
for i, med in enumerate(medians, start=1):
    plt.text(i + 0.1, med*1.3, f"{med:.2f}", ha='left', va='center', color='black')

ax = plt.gca()
comp_idx = [("CPU", "GPU w/o Reg", 2, 0),
            ("CPU", "GPU", 2, 1),
            ("GPU w/o Reg", "GPU", 0, 1)]
add_pval_annotations(ax, correlators, comp_idx, df_wilcoxon, y_offset=0.1, metric="Correlation")

plt.ylabel("Correlation")
plt.ylim(-1, 1.2)  # give space for annotations
plt.savefig("LEMBAS/data_to_report/figs/with_state_reg_comparecorrelators_boxplot_with_p.png", dpi=300, bbox_inches="tight")
plt.close()


# Example for time plot
plt.figure(figsize=(3, 3))
for i in range(3):
    y = times[i]
    x = np.random.normal(i + 1, 0, size=len(y))
    plt.scatter(x, y, alpha=0.7, label=labels[i], color=color_list[i])

# Make boxplot and get its stats
box = plt.boxplot(times, labels=labels, showfliers=False,
                  medianprops=dict(color='black'))

# Add medians as text
medians = [np.median(vals) for vals in times]
for i, med in enumerate(medians, start=1):
    plt.text(i , med*1.2, f"{med:.0f}", ha='left', va='center', color='black')

ax = plt.gca()
comp_idx = [("CPU", "GPU w/o Reg", 2, 0),
            ("CPU", "GPU", 2, 1),
            ("GPU w/o Reg", "GPU", 0, 1)]
add_pval_annotations(ax, times, comp_idx, df_wilcoxon, y_offset=0.08, metric="Execution Time")

plt.ylabel("Time (s)")
plt.ylim(0, 1800)  # extend for annotations
plt.savefig("LEMBAS/data_to_report/figs/with_state_reg_compareTime_boxplot_with_p.png", dpi=300, bbox_inches="tight")
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
plt.tight_layout()
plt.savefig("LEMBAS/data_to_report/figs/with_state_reg_comparetable_1and2.png", dpi=300, bbox_inches='tight')
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

plt.tight_layout()
plt.savefig("LEMBAS/data_to_report/figs/with_state_reg_comparetable_3.png", dpi=300, bbox_inches='tight')
plt.close()

# Optionally, save results
df_ttest.to_csv("ttest_results.csv", index=False) 