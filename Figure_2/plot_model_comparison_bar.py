import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl

# === Make PDF text editable in Illustrator ===
mpl.rcParams['pdf.fonttype'] = 42 
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'  

# === Correlation data ===
#Note, code to implement models and obtain pearson r values are available in SI figures. Model predictions were run on a GPU
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
