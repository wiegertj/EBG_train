import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df_ebg_prediction = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/ebg_prediction_test.csv"))
df_ebg_prediction["dataset"] = df_ebg_prediction["dataset"].str.replace(".csv", "")
df_ebg_prediction["prediction_ebg_tool"] = df_ebg_prediction["prediction_median"]
df_ground_truth = pd.read_csv(os.path.join(os.pardir, "data/processed/target/target.csv"))
df_merged = df_ebg_prediction.merge(df_ground_truth, on=["dataset", "branchId"])
df_merged["support"] = df_merged["support"] * 100

correlation_df = df_merged.groupby('dataset').apply(lambda x: pd.Series({
    'correlation': round(pearsonr(x['prediction_ebg_tool'], x['support'])[0], 2),
    'p_value': round(pearsonr(x['prediction_ebg_tool'], x['support'])[1], 3)
})).reset_index()
correlation_df.to_csv("correlation_results.csv", index=False)

plt.hist(correlation_df['correlation'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Correlation', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("paper_figures/ebg_correlation_histogram.png")