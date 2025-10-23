import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict
import string
from scipy.stats import ks_2samp

root = "LEMBAS/data_to_report/fig_2_3/"
folder_types = ["ligand", "mac", "syn"]
folder_numbers = ["1", "10", "100"]

# editable row labels (change these)
row_labels = ["High-coverage", "Low-coverage", "Synthetic data"]
L2_labels = ["Low", "Medium", "High"]

def load_index_pair(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return np.array(obj[0]), np.array(obj[1])
    if isinstance(obj, dict):
        for kpair in [("0", "1"), ("indices",), ("stochastic_edges",)]:
            if all(k in obj for k in kpair if isinstance(k, str)):
                v = obj[kpair[0]]
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    return np.array(v[0]), np.array(v[1])
        vals = list(obj.values())
        if len(vals) >= 2:
            return np.array(vals[0]), np.array(vals[1])
    arr = np.array(obj)
    if arr.ndim == 2 and arr.shape[0] >= 2:
        return arr[0].astype(int), arr[1].astype(int)
    raise ValueError(f"Can't interpret index pair from {path!r}")

# --- Only 3 rows (CDFs) now ---
fig, axes = plt.subplots(3, len(folder_numbers), figsize=(18, 12),
                         gridspec_kw={'left': 0.20, 'right': 0.98, 'top': 0.95, 'bottom': 0.05,
                                      'hspace': 0.18, 'wspace': 0.15})

# --- CDF plots only ---
for i, ftype in enumerate(folder_types):
    gt_file = os.path.join(root, f"ground_truth/ground_truth_{ftype}.txt")
    true_edges = np.loadtxt(gt_file, delimiter=",", dtype=int)

    for j, fnum in enumerate(folder_numbers):
        folder_path = os.path.join(root, f"parameter_study_{ftype}_{fnum}")
        model_files = glob.glob(os.path.join(folder_path, "*_model_*.pth"))

        def extract_time(fname):
            try:
                return int(os.path.basename(fname).split("_model_")[-1].split(".")[0])
            except:
                return 0

        model_files = sorted(model_files, key=extract_time)
        time_to_files = defaultdict(list)
        for f in model_files:
            t = extract_time(f)
            time_to_files[t].append(f)
        timepoints = sorted(time_to_files.keys())
        if not timepoints:
            continue

        mid_time = timepoints[len(timepoints) // 2]
        mid_files = time_to_files[mid_time]

        added_weights, true_weights = [], []
        for model_file in mid_files:
            model_name = os.path.basename(model_file).split("_model_")[0]
            data = torch.load(model_file, map_location="cpu")
            weights = np.abs(np.array(data['signaling_network.weights'].cpu()))

            stochastic_path = os.path.join(folder_path, f"{model_name}.pth")
            try:
                s0, s1 = load_index_pair(stochastic_path)
            except Exception:
                continue

            added_weights.append(weights[s0, s1])
            true_weights.append(weights[true_edges[0], true_edges[1]])

        if len(added_weights) == 0 or len(true_weights) == 0:
            continue

        added_weights = np.concatenate(added_weights)
        true_weights = np.concatenate(true_weights)

        added_sorted = np.sort(added_weights)
        added_cdf = np.arange(1, len(added_sorted) + 1) / len(added_sorted)
        true_sorted = np.sort(true_weights)
        true_cdf = np.arange(1, len(true_sorted) + 1) / len(true_sorted)

        common_x = np.unique(np.concatenate([added_sorted, true_sorted]))
        added_interp = np.interp(common_x, added_sorted, added_cdf)
        true_interp = np.interp(common_x, true_sorted, true_cdf)

        ax = axes[i, j]
        ax.plot(added_sorted, added_cdf, color='black')
        ax.plot(true_sorted, true_cdf, color='red')
        ax.set_xscale('log')
        ax.set_xlabel("|Weight|", fontsize=20)
        ax.set_ylabel("CDF", fontsize=20)

        ax.fill_between(common_x, added_interp, true_interp, 
                        where=(added_interp > true_interp), 
                        color='grey', alpha=0.2, interpolate=True)
        ax.fill_between(common_x, added_interp, true_interp, 
                        where=(added_interp < true_interp), 
                        color='red', alpha=0.2, interpolate=True)

        ks_stat, p_value = ks_2samp(added_weights, true_weights)
        print(f"KS test for {ftype} folder {fnum}: statistic={ks_stat:.4f}, p-value={p_value:.4e}")
        ax.text(0.05, 0.95, f"p={p_value:.2e}", transform=ax.transAxes,
                ha='left', va='top', fontsize=16)

        if ftype == "syn":
            true_weights = np.load("LEMBAS/data_to_report/fig_2_3/ground_truth/true_network_weights.npy")
            true_weights = np.abs(true_weights[true_edges[0], true_edges[1]])
            true_sorted = np.sort(true_weights)
            true_cdf = np.arange(1, len(true_sorted) + 1) / len(true_sorted)
            # ax.plot(true_sorted, true_cdf, color='black', label="True Weights")
# --- Axis formatting ---
cdf_xmin, cdf_xmax = 1e-7, 5
for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xscale('log')
        ax.set_xlim(cdf_xmin, cdf_xmax)
        if i != 2:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        if j != 0:
            ax.set_yticklabels([])
            ax.set_ylabel('')

# --- Add legend to top-right subplot ---
ax_legend = axes[0, 0]  # top row, rightmost column
ax_legend.plot([], [], color='black', label='Added weights')
ax_legend.plot([], [], color='red', label='PKN weights')
ax_legend.legend(fontsize=16, loc='center left')

# --- Row labels ---
for row_idx, label in enumerate(row_labels):
    ax_left = axes[row_idx, 0]
    pos = ax_left.get_position()
    y_center = 0.5 * (pos.y0 + pos.y1)
    x_left = pos.x0 - 0.05
    fig.text(x_left, y_center, label, va="center", ha="right", fontsize=20, rotation=90)

# --- Add L2 norm titles on top row ---
for j, fnum in enumerate(folder_numbers):
    w_val = 1e-6 * int(fnum)
    axes[0, j].set_title(rf"{L2_labels[j]} $L_2$ ($w_{{L_2}} = {w_val:.1e}$)", fontsize=20, pad=20)
    
plt.savefig(os.path.join(root, "figs/parameter_study_3x3_CDFs.svg"), bbox_inches='tight')
# plt.show()