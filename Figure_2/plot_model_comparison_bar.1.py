import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import numpy as np 
# === Make PDF text editable in Illustrator ===
mpl.rcParams['pdf.fonttype'] = 42   # Embed TrueType fonts, not paths
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text editable if you also save SVGs

# === Correlation data ===
data = pd.DataFrame({
    'Model': ['MMSplice', 'HAL', 'Pangolin', 'SpliceAI'],
    'r': [0.66, 0.67, 0.75, 0.77],
    'n': [70544, 22305, 70843, 70843]
})

# === Plasma colors ===
plasma_colors = sns.color_palette("plasma_r", n_colors=len(data))

# === Barplot ===
plt.figure(figsize=(4, 5))
sns.barplot(data=data, x='Model', y='r', palette=plasma_colors, edgecolor='black')

# === Annotate r and n values ===
for i, row in data.iterrows():
    label = f"r = {row['r']:.2f}\nn = {row['n']:,}"
    plt.text(i, row['r'] + 0.01, label, ha='center', va='bottom', fontsize=12)

# === Formatting ===
plt.ylim(0, 1)
plt.ylabel("Pearson r")
plt.title("Model Correlation with Experimental Δlogit(PSI)")

# === Save as PDF with editable text ===
plt.savefig("/ESL/Figures_SK/Spliceai/model_r_comparison.pdf")



#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import numpy as np

# === Make PDF text editable in Illustrator ===
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === Correlation data ===
data = pd.DataFrame({
    'Set': ['Aggregate', 'Aggregate', 'Aggregate',
            'Test set', 'Test set', 'Test set'],
    'Model': ['Baseline', 'Retrained', 'SpliceAI',
              'Baseline', 'Retrained', 'SpliceAI'],
    'r': [0.66, 0.77, 0.78,
          0.66, 0.69, 0.77],
    'n': [70544, 70544, 70544,
          3526, 3526, 3526]
})

# === Colors: Baseline = orange-gold, Retrained = blue, SpliceAI = purple ===
color_map = {
    'Baseline': '#e4a44e',   # orange-gold
    'Retrained': '#377eb8',  # blue
    'SpliceAI': '#732c8c'    # purple
}

# === Setup grouped bar plot ===
fig, ax = plt.subplots(figsize=(6, 5))

x_labels = ['Aggregate', 'Test set']
x = np.arange(len(x_labels))  # positions for groups
width = 0.25  # width of each bar

# Extract values for plotting
vals = {}
ns = {}
for model in ['Baseline', 'Retrained', 'SpliceAI']:
    vals[model] = [data[(data['Set'] == s) & (data['Model'] == model)]['r'].values[0] for s in x_labels]
    ns[model]   = [data[(data['Set'] == s) & (data['Model'] == model)]['n'].values[0] for s in x_labels]

# Plot bars
rects = {}
rects['Baseline'] = ax.bar(x - width, vals['Baseline'], width, label='Baseline',
                           color=color_map['Baseline'], edgecolor='black')
rects['Retrained'] = ax.bar(x, vals['Retrained'], width, label='Retrained',
                            color=color_map['Retrained'], edgecolor='black')
rects['SpliceAI'] = ax.bar(x + width, vals['SpliceAI'], width, label='SpliceAI',
                           color=color_map['SpliceAI'], edgecolor='black')

# === Annotate r and n values ===
for model in ['Baseline', 'Retrained', 'SpliceAI']:
    for rect, r_val, n_val in zip(rects[model], vals[model], ns[model]):
        ax.text(rect.get_x() + rect.get_width()/2, r_val + 0.015,
                f"r={r_val:.2f}\nn={n_val:,}", ha='center', va='bottom', fontsize=10)

# === Formatting ===
ax.set_ylim(0, 1)
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("Baseline vs Retrained MMSplice vs SpliceAI Correlation", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=12)

# Add legend
ax.legend(frameon=False, fontsize=11, loc="upper left")

# === Save as PDF with editable text ===
plt.tight_layout()
plt.savefig("/ESL/ESL_MPRA/Figure_2/mmsplice_retrain_plots/baseline_retrained_spliceai.pdf")
plt.close()



