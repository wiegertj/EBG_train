import pandas as pd
import matplotlib.pyplot as plt
import os

df_ebg = pd.read_csv(os.pardir + "/data/comparison/simulations/ebg_simulation.csv")
df_alrt = pd.read_csv(os.pardir + "/data/comparison/simulations/alrt_simulation.csv")
df_ufb = pd.read_csv(os.pardir + "/data/comparison/simulations/ufb_simulation.csv")
diffs = pd.read_csv(os.pardir + "/data/comparison/simulations/simu_diffs.csv")


def calculate_balanced_accuracy(group):
    TP = group['Eval'].eq('TP').sum()
    TN = group['Eval'].eq('TN').sum()
    FP = group['Eval'].eq('FP').sum()
    FN = group['Eval'].eq('FN').sum()

    sensitivity_denominator = (TP + FN)
    if sensitivity_denominator == 0:
        sensitivity = 0
    else:
        sensitivity = TP / sensitivity_denominator

    specificity_denominator = (TN + FP)
    if specificity_denominator == 0:
        specificity = 0
    else:
        specificity = TN / specificity_denominator

    balanced_accuracy = (sensitivity + specificity) / 2
    return balanced_accuracy


ebg_balanced_accuracy = df_ebg.groupby('dataset').apply(calculate_balanced_accuracy).reset_index(
    name='balanced_accuracy_EBG')
alrt_balanced_accuracy = df_alrt.groupby('dataset').apply(calculate_balanced_accuracy).reset_index(
    name='balanced_accuracy_SHaLRT')
ufb_balanced_accuracy = df_ufb.groupby('dataset').apply(calculate_balanced_accuracy).reset_index(
    name='balanced_accuracy_UFB')

print("Mean Balanced Accuracy (EBG):", ebg_balanced_accuracy['balanced_accuracy_EBG'].mean())
print("Mean Balanced Accuracy (SHaLRT):", alrt_balanced_accuracy['balanced_accuracy_SHaLRT'].mean())
print("Mean Balanced Accuracy (UFB):", ufb_balanced_accuracy['balanced_accuracy_UFB'].mean())

merged_ebg = pd.merge(diffs, ebg_balanced_accuracy, on='dataset').sort_values(by='diff')
merged_alrt = pd.merge(diffs, alrt_balanced_accuracy, on='dataset').sort_values(by='diff')
merged_ufb = pd.merge(diffs, ufb_balanced_accuracy, on='dataset').sort_values(by='diff')

merged_alrt['running_avg_alrt'] = merged_alrt['balanced_accuracy_SHaLRT'].rolling(window=40, min_periods=10).mean()
merged_ebg['running_avg_ebg'] = merged_ebg['balanced_accuracy_EBG'].rolling(window=40, min_periods=10).mean()
merged_ufb['running_avg_ufb'] = merged_ufb['balanced_accuracy_UFB'].rolling(window=40, min_periods=10).mean()

plt.plot(merged_alrt["diff"], merged_alrt['running_avg_alrt'], label='SH-aLRT')
plt.plot(merged_ebg["diff"], merged_ebg['running_avg_ebg'], label='EBG')
plt.plot(merged_ufb["diff"], merged_ufb['running_avg_ufb'], label='UFBoot2')

plt.xlabel('Difficulty', fontsize=14)
plt.ylabel('Balanced accuracy', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/simulated_data_difficulty.png")
