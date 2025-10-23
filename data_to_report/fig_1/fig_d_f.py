import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu


def load_csv(file_path):
    return pd.read_csv(file_path)


def cohens_d(x, y):
    """Compute Cohen's d for independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def plot_and_save(first_csv_path, second_csv_path,
                  scatter_output_path="LEMBAS/data_to_report/figs/ligand_scatter_plot_16000.svg",
                  boxplot_output_path="LEMBAS/data_to_report/figs/ligand_boxplot_16000.svg",
                  timeplot_output_path="LEMBAS/data_to_report/figs/ligand_time_boxplot_16000.svg"):
    
    df1 = load_csv(first_csv_path)
    df2 = load_csv(second_csv_path)

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(3, 2))
    plt.scatter(df1["Train"], df1["Test"], alpha=0.7, label="Train vs Test (CPU)", color="blue")
    plt.scatter(df2["correlator_train"], df2["correlator_test"], alpha=0.7, label="Train vs Test (GPU)", color="red")

    print("CPU Mean Test:", np.mean(df1["Test"]))
    print("GPU Mean Test:", np.mean(df2["correlator_test"]))

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Train Correlation")
    plt.ylabel("Test Correlation")
    plt.legend()
    fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
    plt.savefig(scatter_output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Boxplot + stats
    test_data = [df1["Test"], df2["correlator_test"]]
    labels = ["CPU", "GPU"]

    fig, ax = plt.subplots(figsize=(3, 2))
    plt.boxplot(test_data, labels=labels, showfliers=True, medianprops=dict(color='black'))

    for i, data_group in enumerate(test_data):
        x = np.random.normal(i + 1, 0.01, size=len(data_group))
        plt.scatter(x, data_group, alpha=0.7)

    for i, data_group in enumerate(test_data):
        median = np.median(data_group)
        plt.text(i + 1 + 0.3, median - 0.05, f'{median:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    try:
        common_len = min(len(df1["Test"]), len(df2["correlator_test"]))
        stat, p = wilcoxon(df1["Test"][:common_len], df2["correlator_test"][:common_len])
        plt.text(1.5, 0.1, f'Wilcoxon p={p:.3f}', ha='center', fontsize=11, color='black')
    except ValueError as e:
        print("Wilcoxon test failed:", e)

    try:
        d = cohens_d(df2["correlator_test"], df1["Test"])
        plt.text(1.5, -0.1, f"Cohen's d={d:.3f}", ha='center', fontsize=11, color='black')
    except Exception as e:
        print("Cohen's d computation failed:", e)

    plt.ylim(-0.2, 1.2)
    plt.ylabel("Correlation")
    fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
    plt.savefig(boxplot_output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Time comparison (stripplot) with Wilcoxon
    df3 = load_csv('LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/CPU_ligand_time.csv')

    fig, ax = plt.subplots(figsize=(3, 2))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    sns.stripplot(y=df3['Time'], x=[0]*len(df3['Time']), ax=ax, alpha=0.7, color=colors[0], size=5, jitter=True, zorder=1)
    sns.stripplot(y=df2['times'], x=[1]*len(df2['times']), ax=ax, alpha=0.7, color=colors[1], zorder=2)

    gpu_mean = df2['times'].mean()
    cpu_mean = df3['Time'].mean()

    ax.hlines(y=cpu_mean, xmin=-0.2, xmax=0.2, colors='black', linestyles='--', linewidth=2)
    ax.hlines(y=gpu_mean, xmin=0.8, xmax=1.2, colors='black', linestyles='--', linewidth=2)

    ax.text(0 + 0.3, cpu_mean + cpu_mean * 0.06, f'{cpu_mean:.0f}', color='black', ha='center')
    ax.text(1 + 0.3, gpu_mean + gpu_mean * 0.12, f'{gpu_mean:.0f}', color='black', ha='center')

    try:
        stat, p = mannwhitneyu(df2['times'], df3['Time'], alternative='two-sided')
        ax.text(0.5, -0, f'M-W U p={p:.3f}', ha='center', fontsize=11, color='black')
    except ValueError as e:
        print("Mann-Whitney U test on times failed:", e)

    # Speed-up calculation
    speed_up = cpu_mean / gpu_mean if gpu_mean != 0 else np.nan
    ax.text(0.5, -1500, f'Speed-up = {speed_up:.2f}x', ha='center', fontsize=11, color='black')
    # Summary Statistics Table
    gpu_corr_mean = np.mean(df2["correlator_test"])
    cpu_corr_mean = np.mean(df1["Test"])

    gpu_time_mean = np.mean(df2['times'][:3])
    cpu_time_mean = np.mean(df3['Time'])


    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CPU', 'GPU'])
    ax.set_xlim(-0.5, 1.5)
    plt.ylim(bottom=-2000, top=12000)
    plt.ylabel("Time (s)")
    fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
    plt.savefig(timeplot_output_path, dpi=300, bbox_inches='tight')
    plt.close()



    speed_up_summary = cpu_time_mean / gpu_time_mean if gpu_time_mean != 0 else np.nan

    summary_stats = {
        "Method": ["GPU", "CPU"],
        "Mean Time (s)": [f"{gpu_time_mean:.4f}", f"{cpu_time_mean:.4f}"],
        "Mean Correlation": [f"{gpu_corr_mean:.4f}", f"{cpu_corr_mean:.4f}"],
        "Speed-up (CPU/GPU)": [f"{speed_up_summary:.2f}x", ""]
    }
    df_summary = pd.DataFrame(summary_stats)

    summary_csv_path = "LEMBAS/data_to_report/figs/ligand_summary_stats_16000.csv"
    df_summary.to_csv(summary_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('off')
    table = ax.table(cellText=df_summary.values,
                     colLabels=df_summary.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)

    summary_table_path = "LEMBAS/data_to_report/figs/ligand_summary_stats_16000.png"
    plt.savefig(summary_table_path, dpi=300, bbox_inches='tight')
    plt.close()
plot_and_save(
    "LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/data_from_orginal.csv",
    "LEMBAS/data_to_report/data_and_anlysis_from_LOOCV/GPU_16000.csv"
)