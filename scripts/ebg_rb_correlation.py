import os
import pandas as pd
from scipy.stats import pearsonr

df_ebg_prediction = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/ebg_prediction_test.csv"))
df_ebg_prediction["dataset"] = df_ebg_prediction["dataset"].str.replace(".csv", "")
df_ebg_prediction["prediction_ebg_tool"] = df_ebg_prediction["prediction_median"]
df_ground_truth = pd.read_csv(os.path.join(os.pardir, "data/processed/target/target.csv"))
df_merged = df_ebg_prediction.merge(df_ground_truth, on=["dataset", "branchId"])
df_merged["support"] = df_merged["support"] * 100

correlation_df = df_merged.groupby('dataset').apply(lambda x: pd.Series({
    'correlation': round(pearsonr(x['prediction_ebg_tool'], x['support'])[0], 2),
    'p_value': round(pearsonr(x['prediction_ebg_tool'], x['support'])[1], 3),
    'mean_support': x['support'].mean()
})).reset_index()
correlation_df.to_csv("correlation_results.csv", index=False)

mean_corr = correlation_df["correlation"].mean()
std_corr = correlation_df["correlation"].std()
print(f"Mean Pearson correlation: {mean_corr}, Std.: {std_corr}")

df_msa = pd.read_csv(os.path.join(os.pardir, "data/processed/msa_difficulty.csv"))
df_merged = correlation_df.merge(df_msa, how="inner", on=["dataset"])
filtered_rows = df_merged[df_merged['correlation'] <= 0.7]
print(filtered_rows[['dataset','difficulty', 'mean_support', 'correlation']])
msa_feats = pd.read_csv("/Users/juliuswiegert/Repositories/placement_difficulty_prediction/data/processed/features/msa_features.csv")
df_merged = df_merged.merge(msa_feats, how="inner", on=["dataset"])

print(df_merged.shape)
