import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import tqdm as tqdm
import matplotlib.colors as mcolors
root = "LEMBAS/data_to_report/fig_2_3/"
folder_numbers = ["1", "10", "100"]
L2_labels = ["Low", "Medium", "High"]
ftypes = ["syn", "mac", "ligand"]


# --- collect all means ---
mean_added = np.zeros((len(ftypes), len(folder_numbers)))
mean_true = np.zeros((len(ftypes), len(folder_numbers)))

for i, ftype in (enumerate(ftypes)):
    gt_file = os.path.join(root, f"ground_truth/ground_truth_{ftype}.txt")
    true_edges = np.loadtxt(gt_file, delimiter=",", dtype=int)

    for j, fnum in (enumerate(folder_numbers)):

        print(i,j)
        folder_path = os.path.join(root, f"parameter_study_{ftype}_{fnum}")
        model_files = glob.glob(os.path.join(folder_path, "*_model_*.pth"))

        def extract_time(fname):
            try:
                return int(os.path.basename(fname).split("_model_")[-1].split(".")[0])
            except:
                return 0

        model_files = sorted(model_files, key=extract_time)
        added_all, true_all = [], []

        for model_file in model_files:
            model_name = os.path.basename(model_file).split("_model_")[0]
            data = torch.load(model_file, map_location="cpu")
            weights = np.array(data['signaling_network.weights'].cpu())
            non_zero_weights = weights[weights != 0]
            if len(non_zero_weights) == 0:
                continue

            median_val = np.median(non_zero_weights)
            stochastic_edges_file = os.path.join(folder_path, f"{model_name}.pth")

            s0, s1=torch.load(stochastic_edges_file, map_location="cpu",weights_only=False)




            added_vals = weights[s0, s1]
            added_below = np.sum(added_vals < median_val) / len(added_vals) * 100

            true_vals = weights[true_edges[0], true_edges[1]]
            true_below = np.sum(true_vals < median_val) / len(true_vals) * 100

            added_all.append(added_below)
            true_all.append(true_below)

        mean_added[i, j] = np.mean(added_all) if added_all else np.nan
        #mean_true[i, j] = np.mean(true_all) if true_all else np.nan


# --- plot heatmap ---
fig, ax = plt.subplots(figsize=(4, 4))

# Define a continuous blue-white-red colormap centered at 50%
cmap = plt.get_cmap("bwr")
norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)

im = ax.imshow(mean_added, cmap=cmap, norm=norm)

# Add annotations
for i in range(mean_added.shape[0]):
    for j in range(mean_added.shape[1]):
        val = mean_added[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=12)
ftypes = ["SYN", "LC", "HC"]
# Set labels and title
ax.set_xticks(range(len(L2_labels)))
ax.set_xticklabels(L2_labels, fontsize=12)
ax.set_yticks(range(len(ftypes)))
ax.set_yticklabels(ftypes, fontsize=12)
#ax.set_title("Added Weights", fontsize=12)

# Add colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Stocastic Edges below median (%)", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(root, "figs/parameter_study_heatmaps_bwr.svg"), bbox_inches='tight')
plt.show()