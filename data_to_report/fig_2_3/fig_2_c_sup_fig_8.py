import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# --- load data ---
data_conditions = ["ligand", "mac", "syn"]
data_set_name_conditions = ["High-coverage", "Low-coverage", "Synthetic"]
L2_condition = [1, 10, 100]
root = Path("LEMBAS/data_to_report/parameter_study")

dfs = {}

for condition in data_conditions:
    for l2 in L2_condition:
        directory = root / f"{condition}_{l2}_training"

        test_file = next(directory.glob("mean_test*.csv"), None)
        train_file = next(directory.glob("mean_train*.csv"), None)

        key = f"{condition}_{l2}"
        dfs[key] = {
            "test": pd.read_csv(test_file) if test_file else None,
            "train": pd.read_csv(train_file) if train_file else None,
        }


# --- plotting ---
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)

for i, condition in enumerate(data_conditions):
    for j, l2 in enumerate(L2_condition):
        ax = axes[i, j]
        key = f"{condition}_{l2}"
        data = dfs[key]

        # Add L2 header only for top row
        if i == 0:

            w_val = 1e-6 * int(l2)
            ax.set_title(rf" $L_2$ ($w_{{L_2}}={w_val:.1e}$)", fontsize=14)
        for label, color in [("train", "red"), ("test", "blue")]:
            if data[label] is not None:
                mean_curve = data[label].mean(axis=0)
                x = np.linspace(0, 100, mean_curve.shape[0])
                ax.plot(x, mean_curve, color=color, label=label.capitalize()+" loss")

                min_idx = np.argmin(mean_curve)
                min_x = x[min_idx]
                min_y = mean_curve[min_idx]
                ax.scatter(min_x, min_y, color=color, s=40, zorder=5)

                # Offset the label and add an arrow pointing to the min point
                x_offset = -5.0
                y_offset = min_y * 1.7
                ax.annotate(
                    f"{min_y:.3f}",
                    xy=(min_x, min_y),
                    xytext=(min_x + x_offset, y_offset),
                    arrowprops=dict(facecolor=color, arrowstyle="->", lw=1.5),
                    fontsize=12,
                    ha='center',
                    va='bottom',
                    color=color
                )
        if i == 2:
            ax.set_xlabel("% of training", fontsize=12)
        if j == 0:
            ax.set_ylabel(data_set_name_conditions[i], fontsize=12)
            ax.set_yscale("log")

# Add a single "Loss" label to the right of dataset names
fig.text(0.04, 0.2, 'Loss', va='center', rotation='vertical', fontsize=12)
fig.text(0.04, 0.515, 'Loss', va='center', rotation='vertical', fontsize=12)
fig.text(0.04, 0.825, 'Loss', va='center', rotation='vertical', fontsize=12)

# show legend only on top-right subplot
axes[0, 2].legend(loc="upper right")

plt.tight_layout()
plt.savefig("LEMBAS/data_to_report/fig_2_3/figs/training_curves.svg", bbox_inches='tight')
#plt.show()
import seaborn as sns

# --- prepare heatmap data ---
min_losses_train = np.zeros((len(data_conditions), len(L2_condition)))
min_losses_test = np.zeros((len(data_conditions), len(L2_condition)))

for i, condition in enumerate(data_conditions):
    for j, l2 in enumerate(L2_condition):
        key = f"{condition}_{l2}"
        data = dfs[key]

        if data["train"] is not None:
            middle_idx = data["train"].shape[1] // 2
            min_losses_train[i, j] = data["train"].mean(axis=0)[-1]
        else:
            min_losses_train[i, j] = np.nan

        if data["test"] is not None:
            middle_idx = data["test"].shape[1] // 2
            min_losses_test[i, j] = data["test"].mean(axis=0)[-1]
        else:
            min_losses_test[i, j] = np.nan
plt.figure(figsize=(4, 4))
data_set_name_conditions = ["HC", "LC", "SYN"]
ax = sns.heatmap(
    min_losses_test,
    annot=True,
    fmt=".3f",
    cmap="Reds",
    xticklabels=[f"L2={l}" for l in L2_condition],
    yticklabels=data_set_name_conditions,
    cbar=False,   # enable colorbar
    vmin=0,      # min value for color scale
    vmax=0.05     # max value for color scale
)

# Reverse the y-axis
ax.invert_yaxis()

#plt.title("Minimum Test Loss", fontsize=14)
plt.tight_layout()
plt.savefig("LEMBAS/data_to_report/fig_2_3/figs/last_loss_heatmap.svg", bbox_inches='tight')
#plt.show()
