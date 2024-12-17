import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_ebg_prediction = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/ebg_prediction_test.csv"))
df_ebg_prediction["dataset"] = df_ebg_prediction["dataset"].str.replace(".csv", "")
df_ebg_prediction["prediction_ebg_tool"] = df_ebg_prediction["prediction_median"]
df_raxml = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/raxml_classic_supports.csv"))
df_merged = df_raxml.merge(df_ebg_prediction, on=["dataset", "branchId"], how="inner")
df_iqtree = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/iq_boots.csv"),
                        usecols=lambda column: column != 'Unnamed: 0')
df_merged = df_merged.merge(df_iqtree, left_on=["dataset", "branchId"], right_on=["dataset", "branchId_true"],
                            how="inner")

# Calculate uncertainty
df_merged["bound_dist_5"] = abs(df_merged["prediction_lower5"] - df_merged["prediction_median"])
df_merged["bound_dist_10"] = abs(df_merged["prediction_lower10"] - df_merged["prediction_median"])

# Get abs. prediction error per uncertainty bin and get median error per bin
bins = np.arange(0, 41, 1)
bound_dist_10 = df_merged["bound_dist_10"]
bound_dist_5 = df_merged["bound_dist_5"]
df_merged["pred_error_ebg"] = abs(df_merged["true_support"] - df_merged["prediction_ebg_tool"])
pred_error_ebg = df_merged["pred_error_ebg"]

interval_categories_10 = pd.cut(bound_dist_10, bins)
interval_categories_5 = pd.cut(bound_dist_5, bins)

grouped_10 = pred_error_ebg.groupby(interval_categories_10).median()
grouped_5 = pred_error_ebg.groupby(interval_categories_5).median()

# Create plot
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(grouped_10))

plt.bar(x - bar_width/2, grouped_10.values, width=bar_width, label='10% lower bound')
plt.bar(x + bar_width/2, grouped_5.values, width=bar_width, label='5% lower bound')

plt.xlabel('Distance to median prediction', fontsize=14)
plt.ylabel('Median absolute error', fontsize=14)
plt.grid(axis='y')
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

fig = plt.gcf()
fig.set_size_inches(10, 6)

plt.tight_layout()
plt.savefig("paper_figures/ebg_uncertainty_filter_regressor_high.png", dpi=900)
