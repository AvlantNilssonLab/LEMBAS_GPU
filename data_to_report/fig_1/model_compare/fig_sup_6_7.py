import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load csv
df = pd.read_csv("LEMBAS/data_to_report/fig_1/model_compare/input_data_XX_YY_comparison.csv")

# filter only rows with "A" in Comparison
df_A = df[df['Comparison'].str.contains("A")]

# choose which weight column to analyze
weight_col = "signaling_weights"

# sort conditions alphabetically
conditions = sorted(df_A['Comparison'].unique())
df_A['Comparison'] = pd.Categorical(df_A['Comparison'], categories=conditions, ordered=True)

# group values per condition
grouped = [df_A.loc[df_A['Comparison'] == cond, weight_col].values for cond in conditions]
scale=2.3
plt.figure(figsize=(2/scale, 2.5/scale))
blue_color = "#1f77b4"

# scatter (jittered samples)
for i, cond in enumerate(conditions, start=1):
    y = df_A.loc[df_A['Comparison'] == cond, weight_col].values
    x = np.random.normal(i, 0.04, size=len(y))  # jitter
    plt.scatter(y, x, alpha=0.7, color=blue_color, s=15/scale, zorder=1)  # swap x/y

# horizontal boxplot
plt.boxplot(grouped, labels=conditions, showfliers=False,
            vert=False,  # horizontal orientation
            boxprops=dict(color='black', linewidth=1.0/scale),
            whiskerprops=dict(color='black', linewidth=1.0/scale),
            capprops=dict(color='black', linewidth=1.0/scale),
            medianprops=dict(color='black', linewidth=1.0/scale))

plt.xlabel("Correlation",fontsize=6/scale)
plt.xticks(fontsize=6/scale)
plt.yticks([])   # hides y-axis tick labels
plt.tight_layout()
plt.xlim(0.0, 1.0)  # swap from ylim to xlim

# make the surrounding axes box thicker
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.savefig('LEMBAS/data_to_report/fig_1/model_compare/XX_compared_YY_horizontal.png',
            dpi=300, bbox_inches='tight')

# load csv
df = pd.read_csv("LEMBAS/data_to_report/fig_1/model_compare/input_data_XX_YY_comparison.csv")


# choose which weight column to analyze
weight_col = "signaling_weights"

# sort conditions alphabetically
conditions = sorted(df['Comparison'].unique())
df['Comparison'] = pd.Categorical(df['Comparison'], categories=conditions, ordered=True)

# group values per condition
grouped = [df.loc[df['Comparison'] == cond, weight_col].values for cond in conditions]

plt.figure(figsize=(6, 2))
blue_color = "#1f77b4"

# scatter (jittered samples)
for i, cond in enumerate(conditions, start=1):
    y = df.loc[df['Comparison'] == cond, weight_col].values
    x = np.random.normal(i, 0.04, size=len(y))  # jitter
    plt.scatter(x, y, alpha=0.7, color=blue_color, s=30, zorder=1)

# boxplot on top (transparent fill, thick black lines)
plt.boxplot(grouped, labels=conditions, showfliers=False,
            boxprops=dict(color='black', linewidth=1.0),
            whiskerprops=dict(color='black', linewidth=1.0),
            capprops=dict(color='black', linewidth=1.0),
            medianprops=dict(color='black', linewidth=1.0))


plt.ylabel("Correlation")
#plt.xticks(rotation=45)
plt.tight_layout()
plt.ylim(0.1, 0.8)
# make the surrounding axes box thicker
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.savefig('LEMBAS/data_to_report/fig_1/model_compare/XX_compared_YY_all.png', dpi=300, bbox_inches='tight')

# load csv
df = pd.read_csv("LEMBAS/data_to_report/fig_1/model_compare/input_data_XX_YY_comparison.csv")


# choose which weight column to analyze
weight_col = "output_layer_weights"

# sort conditions alphabetically
conditions = sorted(df['Comparison'].unique())
df['Comparison'] = pd.Categorical(df['Comparison'], categories=conditions, ordered=True)

# group values per condition
grouped = [df.loc[df['Comparison'] == cond, weight_col].values for cond in conditions]

plt.figure(figsize=(6, 2))
blue_color = "#1f77b4"

# scatter (jittered samples)
for i, cond in enumerate(conditions, start=1):
    y = df.loc[df['Comparison'] == cond, weight_col].values
    x = np.random.normal(i, 0.04, size=len(y))  # jitter
    plt.scatter(x, y, alpha=0.7, color=blue_color, s=30, zorder=1)

# boxplot on top (transparent fill, thick black lines)
plt.boxplot(grouped, labels=conditions, showfliers=False,
            boxprops=dict(color='black', linewidth=1.0),
            whiskerprops=dict(color='black', linewidth=1.0),
            capprops=dict(color='black', linewidth=1.0),
            medianprops=dict(color='black', linewidth=1.0))


plt.ylabel("Correlation")
#plt.xticks(rotation=45)
plt.tight_layout()
plt.ylim(-1.1, 1.1)
# make the surrounding axes box thicker
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.savefig('LEMBAS/data_to_report/fig_1/model_compare/XX_compared_YY_all_output_layer_weights.png', dpi=300, bbox_inches='tight')

