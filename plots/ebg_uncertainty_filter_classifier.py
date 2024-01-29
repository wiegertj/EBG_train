import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score

df_ebg_prediction = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/ebg_prediction_test.csv"))
df_ebg_prediction["dataset"] = df_ebg_prediction["dataset"].str.replace(".csv", "")
df_ebg_prediction["prediction_ebg_tool"] = df_ebg_prediction["prediction_median"]
df_raxml = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/raxml_classic_supports.csv"))
df_merged = df_raxml.merge(df_ebg_prediction, on=["dataset", "branchId"], how="inner")
df_iqtree = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/iq_boots.csv"),
                        usecols=lambda column: column != 'Unnamed: 0')
df_merged = df_merged.merge(df_iqtree, left_on=["dataset", "branchId"], right_on=["dataset", "branchId_true"],
                            how="inner")

# Compute prediction uncertainty
for index, row in df_merged.iterrows():
    entropy_row = entropy(np.array([row["prediction_bs_over_80"], 1 - row["prediction_bs_over_80"]]),
                          base=2)
    df_merged.loc[index, 'entropy_ebg_over_80'] = entropy_row

for index, row in df_merged.iterrows():
    entropy_row = entropy(np.array([row["prediction_bs_over_85"], 1 - row["prediction_bs_over_85"]]),
                          base=2)
    df_merged.loc[index, 'entropy_ebg_over_85'] = entropy_row

for index, row in df_merged.iterrows():
    entropy_row = entropy(np.array([row["prediction_bs_over_75"], 1 - row["prediction_bs_over_75"]]),
                          base=2)
    df_merged.loc[index, 'entropy_ebg_over_75'] = entropy_row

for index, row in df_merged.iterrows():
    entropy_row = entropy(np.array([row["prediction_bs_over_70"], 1 - row["prediction_bs_over_70"]]),
                          base=2)
    df_merged.loc[index, 'entropy_ebg_over_70'] = entropy_row

df_merged['ebg_over_80'] = (df_merged['prediction_bs_over_80'] >= 0.5).astype(int)
df_merged['ebg_over_70'] = (df_merged['prediction_bs_over_70'] >= 0.5).astype(int)
df_merged['ebg_over_75'] = (df_merged['prediction_bs_over_75'] >= 0.5).astype(int)
df_merged['ebg_over_85'] = (df_merged['prediction_bs_over_85'] >= 0.5).astype(int)

df_merged['support_over_85'] = (df_merged['true_support'] >= 85).astype(int)
df_merged['support_over_80'] = (df_merged['true_support'] >= 80).astype(int)
df_merged['support_over_75'] = (df_merged['true_support'] >= 75).astype(int)
df_merged['support_over_70'] = (df_merged['true_support'] >= 70).astype(int)

intervals = np.arange(0, 1.1, 0.1)

# Calculate accuracies fitlered by uncertainty
fractions_80 = []
for i in range(len(intervals) - 1):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    mask = (df_merged['entropy_ebg_over_80'] >= lower_bound) & (df_merged['entropy_ebg_over_80'] < upper_bound)
    fraction = accuracy_score(df_merged['support_over_80'][mask], df_merged['ebg_over_80'][mask])
    fractions_80.append(fraction)

fractions_85 = []
for i in range(len(intervals) - 1):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    mask = (df_merged['entropy_ebg_over_85'] >= lower_bound) & (df_merged['entropy_ebg_over_85'] < upper_bound)
    fraction = accuracy_score(df_merged['support_over_85'][mask], df_merged['ebg_over_85'][mask])
    fractions_85.append(fraction)

fractions_75 = []
for i in range(len(intervals) - 1):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    mask = (df_merged['entropy_ebg_over_75'] >= lower_bound) & (df_merged['entropy_ebg_over_75'] < upper_bound)
    fraction = accuracy_score(df_merged['support_over_75'][mask], df_merged['ebg_over_75'][mask])
    fractions_75.append(fraction)

fractions_70 = []
for i in range(len(intervals) - 1):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    mask = (df_merged['entropy_ebg_over_70'] >= lower_bound) & (df_merged['entropy_ebg_over_70'] < upper_bound)
    fraction = accuracy_score(df_merged['support_over_70'][mask], df_merged['ebg_over_70'][mask])
    fractions_70.append(fraction)

bar_width = 0.2
uncertainty_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

x = np.arange(len(uncertainty_levels))

plt.bar(x - 1.5 * bar_width, fractions_70, width=bar_width, label='t=70')
plt.bar(x - 0.5 * bar_width, fractions_75, width=bar_width, label='t=75')
plt.bar(x + 0.5 * bar_width, fractions_80, width=bar_width, label='t=80')
plt.bar(x + 1.5 * bar_width, fractions_85, width=bar_width, label='t=85')
plt.axhline(y=0.8, color='black', linestyle='--', label='0.8')
plt.axhline(y=0.9, color='blue', linestyle='--', label='0.9')
plt.xlabel('Uncertainty')
plt.xticks(x, [f"{u:.1f}" for u in uncertainty_levels])

plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1.1, 0.1))

plt.legend()
plt.tight_layout()
plt.savefig("paper_figures/ebg_uncertainty_filter_classifier.png")
