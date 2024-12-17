import pandas as pd
import numpy as np
import os
import math
from scipy.stats import entropy
from sklearn.metrics import mean_absolute_error, median_absolute_error, f1_score, roc_auc_score, \
    balanced_accuracy_score, mean_squared_error


def MBE(y_true, y_pred):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_true - y_pred)
    mbe = diff.mean()
    return mbe


df_ebg_prediction = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/ebg_prediction_test.csv"))
df_ebg_prediction["dataset"] = df_ebg_prediction["dataset"].str.replace(".csv", "")
df_ebg_prediction["prediction_ebg_tool"] = df_ebg_prediction["prediction_median"]
df_raxml = pd.read_csv(os.path.join(os.pardir, "data/comparison/predictions/raxml_classic_supports.csv"))
df_merged = df_raxml.merge(df_ebg_prediction, on=["dataset", "branchId"], how="inner")
df_ground_truth = pd.read_csv(os.path.join(os.pardir, "data/processed/target/target.csv"))
df_merged = df_ebg_prediction.merge(df_ground_truth, on=["dataset", "branchId"])
df_merged["true_support"] = df_merged["support"] * 100
df_merged = df_merged.merge(df_raxml, on=["dataset", "branchId"], how="inner")
print(df_merged.shape)

f_merged = df_merged[(df_merged['true_support'] <= 95) & (df_merged['true_support'] >= 60)]
print(df_merged.shape)

df_merged['ebg_over_80'] = (df_merged['prediction_bs_over_80'] >= 0.5).astype(int)
df_merged['ebg_over_70'] = (df_merged['prediction_bs_over_70'] >= 0.5).astype(int)
df_merged['ebg_over_75'] = (df_merged['prediction_bs_over_75'] >= 0.5).astype(int)
df_merged['ebg_over_85'] = (df_merged['prediction_bs_over_85'] >= 0.5).astype(int)

df_merged['rb_over_80'] = (df_merged['support_raxml_classic'] >= 80).astype(int)
df_merged['rb_over_70'] = (df_merged['support_raxml_classic'] >= 70).astype(int)
df_merged['rb_over_75'] = (df_merged['support_raxml_classic'] >= 75).astype(int)
df_merged['rb_over_85'] = (df_merged['support_raxml_classic'] >= 85).astype(int)

df_merged['support_over_85'] = (df_merged['true_support'] >= 85).astype(int)
df_merged['support_over_80'] = (df_merged['true_support'] >= 80).astype(int)
df_merged['support_over_75'] = (df_merged['true_support'] >= 75).astype(int)
df_merged['support_over_70'] = (df_merged['true_support'] >= 70).astype(int)


df_merged["error"] = df_merged["true_support"] - df_merged["prediction_median"]

mean_error_by_dataset = df_merged.groupby('dataset')['error'].median()
import pandas as pd
# Convert the result back to a DataFrame if needed
mean_error_by_dataset_df = pd.DataFrame(mean_error_by_dataset)
pars_dist = pd.read_csv(os.path.join(os.pardir, "data/rf_pars.csv"))
pars_dist = mean_error_by_dataset_df.merge(pars_dist, on=["dataset"], how="inner")
import matplotlib.pyplot as plt

plt.scatter(pars_dist['rf_pars'], pars_dist['error'], color='blue')
plt.title('Error vs. RF Pars')
plt.xlabel('RF Pars')
plt.ylabel('Error')
plt.grid(True)
plt.show()



print("\n" + "#" * 40 + " EBG (Unfiltered) " + "#" * 40)
mae = mean_absolute_error(df_merged["true_support"], df_merged["prediction_median"])
mdae = median_absolute_error(df_merged["true_support"], df_merged["prediction_median"])
mbe = MBE(df_merged["true_support"], df_merged["prediction_median"])
rmse = math.sqrt(mean_squared_error(df_merged["true_support"], df_merged["prediction_median"]))

print("MAE " + str(mae))
print("MDAE " + str(mdae))
print("MBE " + str(mbe))
print("RMSE " + str(rmse))

print("---------t=80-----------")
accuracy = balanced_accuracy_score(df_merged["support_over_80"], df_merged["ebg_over_80"])
f1 = f1_score(df_merged["support_over_80"], df_merged["ebg_over_80"])
roc = roc_auc_score(df_merged["support_over_80"], df_merged["ebg_over_80"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")
print("---------t=70-----------")
accuracy = balanced_accuracy_score(df_merged["support_over_70"], df_merged["ebg_over_70"])
f1 = f1_score(df_merged["support_over_70"], df_merged["ebg_over_70"])
roc = roc_auc_score(df_merged["support_over_70"], df_merged["ebg_over_70"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")
print("---------t=85-----------")
accuracy = balanced_accuracy_score(df_merged["support_over_85"], df_merged["ebg_over_85"])
f1 = f1_score(df_merged["support_over_85"], df_merged["ebg_over_85"])
roc = roc_auc_score(df_merged["support_over_85"], df_merged["ebg_over_85"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")
print("---------t=75-----------")
accuracy = balanced_accuracy_score(df_merged["support_over_75"], df_merged["ebg_over_75"])
f1 = f1_score(df_merged["support_over_75"], df_merged["ebg_over_75"])
roc = roc_auc_score(df_merged["support_over_75"], df_merged["ebg_over_75"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")

print("\n" + "#" * 20 + " EBG (Regression filter distance median and lower bound 5% <= 23) " + "#" * 20)
df_merged["bound_dist_5"] = abs(df_merged["prediction_lower5"] - df_merged["prediction_median"])
df_merged["bound_dist_10"] = abs(df_merged["prediction_lower10"] - df_merged["prediction_median"])
df_merged_reg = df_merged[df_merged["bound_dist_5"] <= 23]


mae = mean_absolute_error(df_merged_reg["true_support"], df_merged_reg["prediction_median"])
mdae = median_absolute_error(df_merged_reg["true_support"], df_merged_reg["prediction_median"])
mbe = MBE(df_merged_reg["true_support"], df_merged_reg["prediction_median"])
rmse = math.sqrt(mean_squared_error(df_merged_reg["true_support"], df_merged_reg["prediction_median"]))

print("MAE " + str(mae))
print("MDAE " + str(mdae))
print("MBE " + str(mbe))
print("RMSE " + str(rmse))

print("\n" + "#" * 20 + " EBG (Classifcation filter uncertainty <= 0.7) " + "#" * 20)

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

df_merged_class = df_merged[df_merged["entropy_ebg_over_80"] <= 0.7]

print("---------t=80-----------")
accuracy = balanced_accuracy_score(df_merged_class["support_over_80"], df_merged_class["ebg_over_80"])
f1 = f1_score(df_merged_class["support_over_80"], df_merged_class["ebg_over_80"])
roc = roc_auc_score(df_merged_class["support_over_80"], df_merged_class["ebg_over_80"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")
print("---------t=70-----------")
accuracy = balanced_accuracy_score(df_merged_class["support_over_70"], df_merged_class["ebg_over_70"])
f1 = f1_score(df_merged_class["support_over_70"], df_merged_class["ebg_over_70"])
roc = roc_auc_score(df_merged_class["support_over_70"], df_merged_class["ebg_over_70"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")
print("---------t=85-----------")
accuracy = balanced_accuracy_score(df_merged_class["support_over_85"], df_merged_class["ebg_over_85"])
f1 = f1_score(df_merged_class["support_over_85"], df_merged_class["ebg_over_85"])
roc = roc_auc_score(df_merged_class["support_over_85"], df_merged_class["ebg_over_85"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")
print("---------t=75-----------")
accuracy = balanced_accuracy_score(df_merged_class["support_over_75"], df_merged_class["ebg_over_75"])
f1 = f1_score(df_merged_class["support_over_75"], df_merged_class["ebg_over_75"])
roc = roc_auc_score(df_merged_class["support_over_75"], df_merged_class["ebg_over_75"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")

print("\n" + "#" * 20 + " Rapid Bootstrap " + "#" * 20)
mae = mean_absolute_error(df_merged["true_support"], df_merged["support_raxml_classic"])
mdae = median_absolute_error(df_merged["true_support"], df_merged["support_raxml_classic"])
mbe = MBE(df_merged["true_support"], df_merged["support_raxml_classic"])
rmse = math.sqrt(mean_squared_error(df_merged["true_support"], df_merged["support_raxml_classic"]))
print("MAE " + str(mae))
print("MDAE " + str(mdae))
print("MBE " + str(mbe))
print("RMSE " + str(rmse))

print("---------t=80-----------")
accuracy = balanced_accuracy_score(df_merged_class["support_over_80"], df_merged_class["rb_over_80"])
f1 = f1_score(df_merged_class["support_over_80"], df_merged_class["rb_over_80"])
roc = roc_auc_score(df_merged_class["support_over_80"], df_merged_class["rb_over_80"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")
print("---------t=70-----------")
accuracy = balanced_accuracy_score(df_merged_class["support_over_70"], df_merged_class["rb_over_70"])
f1 = f1_score(df_merged_class["support_over_70"], df_merged_class["rb_over_70"])
roc = roc_auc_score(df_merged_class["support_over_70"], df_merged_class["rb_over_70"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")
print("---------t=85-----------")
accuracy = balanced_accuracy_score(df_merged_class["support_over_85"], df_merged_class["rb_over_85"])
f1 = f1_score(df_merged_class["support_over_85"], df_merged_class["rb_over_85"])
roc = roc_auc_score(df_merged_class["support_over_85"], df_merged_class["rb_over_85"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")
print("---------t=75-----------")
accuracy = balanced_accuracy_score(df_merged_class["support_over_75"], df_merged_class["rb_over_75"])
f1 = f1_score(df_merged_class["support_over_75"], df_merged_class["rb_over_75"])
roc = roc_auc_score(df_merged_class["support_over_75"], df_merged_class["rb_over_75"])
print(f"BAC: {accuracy} F1: {f1} AUC: {roc}")