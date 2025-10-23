import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

root = "LEMBAS/data_to_report/fig_2_3/"
folder_numbers = ["1", "10", "100"]
L2_labels = ["Low", "Medium", "High"]

# --- Helper to load index-like files robustly ---
def load_index_pair(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return np.array(obj[0]), np.array(obj[1])
    if isinstance(obj, dict):
        vals = list(obj.values())
        if len(vals) >= 2:
            return np.array(vals[0]), np.array(vals[1])
    arr = np.array(obj)
    if arr.ndim == 2 and arr.shape[0] >= 2:
        return arr[0].astype(int), arr[1].astype(int)
    raise ValueError(f"Can't interpret index pair from {path!r}")

# --- Figure setup ---
fig, axes = plt.subplots(1, len(folder_numbers), figsize=(18, 5),
                         gridspec_kw={'left': 0.1, 'right': 0.98, 'top': 0.9, 'bottom': 0.15, 'wspace': 0.25})

ftype = "mac"
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
    results = {}

    for model_file in model_files:
        model_name = os.path.basename(model_file).split("_model_")[0]
        data = torch.load(model_file, map_location="cpu")
        weights = np.array(data['signaling_network.weights'].cpu())
        non_zero_weights = weights[weights != 0]
        if len(non_zero_weights) == 0:
            continue
        median_val = np.median(non_zero_weights)

        stochastic_edges_file = os.path.join(folder_path, f"{model_name}.pth")
        try:
            s0, s1 = load_index_pair(stochastic_edges_file)
        except Exception:
            continue

        added_vals = weights[s0, s1]
        added_below = np.sum(added_vals < median_val) / len(added_vals) * 100
        true_vals = weights[true_edges[0], true_edges[1]]
        true_below = np.sum(true_vals < median_val) / len(true_vals) * 100

        results.setdefault(model_name, {"added": [], "true": []})
        results[model_name]["added"].append(added_below)
        results[model_name]["true"].append(true_below)

    ax = axes[j]
    for model_name, vals in results.items():
        ax.plot(range(len(vals["added"])), vals["added"], color="grey", alpha=0.3)
        ax.plot(range(len(vals["true"])), vals["true"], color="red", alpha=0.3)

    if len(results) > 0:
        all_added = np.array([vals["added"] for vals in results.values()])
        mean_added = np.mean(all_added, axis=0)
        ax.plot(range(len(mean_added)), mean_added, color="black", linewidth=2)

    if j == len(folder_numbers) - 1 and len(results) > 0:
        ax.plot([], [], color="black", label="Added weights")
        ax.plot([], [], color="red", label="PKN weights")
        ax.legend(fontsize=14, loc="lower right")

    w_val = 1e-6 * int(fnum)
    ax.set_ylim(0, 100)
    ax.set_title(rf"{L2_labels[j]} $L_2$ ($w_{{L_{2}}} = {w_val:.1e}$)", fontsize=16)
    ax.set_xlabel("Checkpoints", fontsize=16)
    ax.set_ylabel("% below median", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

plt.savefig("LEMBAS/data_to_report/figs/parameter_study_mac_1by3.svg", bbox_inches='tight')
# plt.show()
