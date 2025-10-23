import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib.colors import ListedColormap

# --- Figure setup ---
fig = plt.figure(figsize=(12, 3))

ax0_pos = [0.05, 0.1, 0.15, 0.8]
ax1_pos = [0.22, 0.1, 0.2, 0.8]
ax2_pos = [0.45, 0.1, 0.2, 0.8]

ax0 = fig.add_axes(ax0_pos)
ax1 = fig.add_axes(ax1_pos)
ax2 = fig.add_axes(ax2_pos)

# --- Fixed y positions ---
conditions = ["A", "B", "C", "D", "E"]
y_positions = {cond: i for i, cond in enumerate(conditions, start=1)}

# --- Heatmap ---
df_heat = pd.read_csv("LEMBAS/data_to_report/fig_1/model_compare/input_data_XX_YY.csv")

def parse_conditions(s):
    parts = s.split('-')
    out = {}
    for p in parts:
        if '^' in p:
            key, val = p.split('^')
            out[key] = int(val)
    return out

df_conditions = df_heat.iloc[:, 0].apply(parse_conditions)
parsed_df = pd.DataFrame(df_conditions.tolist())
df_heat = pd.concat([df_heat['condition'], parsed_df], axis=1)
df_heat = df_heat.set_index('condition')
df_heat = df_heat.drop('k', axis=1)
df_heat = df_heat.iloc[:5, :]
df_heat = df_heat.loc[conditions]

rename_map = {
    "uniform": "State \n Regular \n -ization",
    "L2_or_rand": "Weight \n Regular \n -ization",
    "ProjectOutput_bias": "Output \n Layer"
}
df_heat_renamed = df_heat.rename(columns=rename_map)
# --- Floating boxes ---
colors_map = {0: "black", 1: "#d3d3d3"}
n_cols = len(df_heat_renamed.columns)

for col_idx, col in enumerate(df_heat_renamed.columns):
    for row_idx, condition in enumerate(reversed(df_heat_renamed.index)):
        value = df_heat_renamed.loc[condition, col]
        alpha=0.3
        rect = Rectangle(
            (col_idx+alpha/2, row_idx+alpha/2),
            width=0.8-alpha, height=0.8-alpha,
            facecolor=colors_map.get(value, "white"),
        )
        ax0.add_patch(rect)

ax0.set_xlim(0, n_cols)
ax0.set_ylim(0, len(conditions))
ax0.set_yticks(np.arange(len(conditions)) + 0.4)
ax0.set_yticklabels(reversed(conditions), fontsize=10)
ax0.set_xticks([])
for spine in ax0.spines.values():
    spine.set_visible(False)
# --- Labels ---
ax0_label = fig.add_axes([ax0_pos[0], ax0_pos[1] + ax0_pos[3], ax0_pos[2], 0.05])
ax0_label.axis('off')

label_texts = [ "State \n Regular \n -ization", "Weight \n Regular \n -ization", "Output \n Layer"]
for i, label in enumerate(label_texts):
    ax0_label.text(
        (i + 0.4) / len(label_texts),
        -18.0,
        label,
        ha='center', va='center',
        fontsize=10
    )

# --- Middle: Correlation scatter/boxplot ---
df_corr = pd.read_csv("LEMBAS/data_to_report/fig_1/model_compare/input_data_XX_YY.csv", index_col=0)
df_corr['condition'] = pd.Categorical(df_corr['condition'], categories=conditions, ordered=True)
grouped = [df_corr.loc[df_corr['condition'] == cond, 'samples_pearson'].values for cond in conditions]

blue_color = "#1f77b4"
for cond in conditions:
    y = np.random.normal(y_positions[cond], 0.04, size=len(df_corr.loc[df_corr['condition'] == cond, 'samples_pearson'].values))
    x = df_corr.loc[df_corr['condition'] == cond, 'samples_pearson'].values
    ax1.scatter(x, y, alpha=0.7, color=blue_color, s=10, zorder=1)

ax1.boxplot(grouped, labels=conditions, vert=False, showfliers=False,
            positions=[y_positions[cond] for cond in conditions],
            boxprops=dict(color='black', linewidth=1.0),
            whiskerprops=dict(color='black', linewidth=1.0),
            capprops=dict(color='black', linewidth=1.0),
            medianprops=dict(color='black', linewidth=1.0))

ax1.set_xlim(-1.0, 1.0)
ax1.set_yticks([])
ax1.set_xlabel("Correlation", fontsize=10)
ax1.set_ylabel('')
ax1.tick_params(axis='x', labelsize=10)
ax1.set_title("Prediction of models LOOCV", fontsize=10)
# FLIP Y-axis
ax1.set_ylim(len(conditions) + 0.5, 0.5)

# --- Right: Comparison scatter/boxplot ---
df_comp = pd.read_csv("LEMBAS/data_to_report/fig_1/model_compare/input_data_XX_YY_comparison.csv")
df_A = df_comp[df_comp['Comparison'].str.contains("A")]
weight_col = "signaling_weights"

conditions_comp = sorted(df_A['Comparison'].unique())
df_A['Comparison'] = pd.Categorical(df_A['Comparison'], categories=conditions_comp, ordered=True)
grouped = [df_A.loc[df_A['Comparison'] == cond, weight_col].values for cond in conditions_comp if cond != "A"]

focus_height = y_positions["B"]
comparison_positions = {cond: y_positions.get(cond, i+2) for i, cond in enumerate(conditions_comp)}
shift = comparison_positions.get("B", 2) - focus_height

for cond in conditions_comp:
    if cond == "A":
        continue
    y = np.random.normal(comparison_positions[cond] - shift, 0.04,
                         size=len(df_A.loc[df_A['Comparison'] == cond, weight_col].values))
    x = df_A.loc[df_A['Comparison'] == cond, weight_col].values
    ax2.scatter(x, y, alpha=0.7, color=blue_color, s=10, zorder=1)

ax2.boxplot(grouped, labels=[cond for cond in conditions_comp if cond != "A"], showfliers=False,
            vert=False,
            positions=[comparison_positions[cond] - shift for cond in conditions_comp if cond != "A"],
            boxprops=dict(color='black', linewidth=1.0),
            whiskerprops=dict(color='black', linewidth=1.0),
            capprops=dict(color='black', linewidth=1.0),
            medianprops=dict(color='black', linewidth=1.0))

ax2.set_xlabel("Correlation", fontsize=10)
ax2.set_xlim(0.0, 1.0)
ax2.set_yticks([])
ax2.set_ylabel('')
ax2.tick_params(axis='x', labelsize=10)
ax2.set_title("Model A weights correlation to model X", fontsize=10)

# FLIP Y-axis
ax2.set_ylim(len(conditions) + 0.5, 0.5)

for spine in ax2.spines.values():
    spine.set_linewidth(1.0)


plt.savefig('LEMBAS/data_to_report/figs/setting_compare.svg',
            dpi=300, bbox_inches='tight')