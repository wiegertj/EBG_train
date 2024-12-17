import sys

import pandas as pd
import matplotlib.pyplot as plt
import os

import pandas as pd

# Load the CSV files into pandas DataFrames
df_ebg = pd.read_csv(os.pardir + "/data/comparison/simulations/ebg_simulation_80_unc.csv")
df_ebg_uncertainty = pd.read_csv(os.pardir + "/data/comparison/simulations/ebg_simulation_80_unc.csv")
df_alrt = pd.read_csv(os.pardir + "/data/comparison/simulations/alrt_simulation.csv")
df_ufb = pd.read_csv(os.pardir + "/data/comparison/simulations/ufb_simulation.csv")
df_sbs = pd.read_csv(os.pardir + "/data/comparison/simulations/sbs_simulation.csv")
df_rb = pd.read_csv(os.pardir + "/data/comparison/simulations/rb_simulation.csv")

def get_common_pairs(dfs, key_columns):
    common_pairs = set.intersection(*[set(df[key_columns].drop_duplicates().apply(tuple, axis=1)) for df in dfs])
    return pd.DataFrame(list(common_pairs), columns=key_columns)

# Get the common (dataset, branchID_True) pairs
common_pairs = get_common_pairs([df_ebg, df_ebg_uncertainty, df_alrt, df_ufb, df_sbs, df_rb], ['dataset', 'branchID_True'])

# Filter each dataframe based on the common pairs
def filter_dataframe(df, common_pairs):
    return df.merge(common_pairs, on=['dataset', 'branchID_True'], how='inner')

df_ebg = filter_dataframe(df_ebg, common_pairs)
df_ebg_uncertainty = filter_dataframe(df_ebg_uncertainty, common_pairs)
df_alrt = filter_dataframe(df_alrt, common_pairs)
df_ufb = filter_dataframe(df_ufb, common_pairs)
df_sbs = filter_dataframe(df_sbs, common_pairs)
df_rb = filter_dataframe(df_rb, common_pairs)

# Define a function to calculate the required measures
def calculate_measures(df, support_column, truth_column, bound=5, filtered = False, ebg=False):
    results = {}

    if ebg:
        print(bound)

        if not filtered:
            df_tmp = df[df[truth_column] == 0]
            type_1_error = (df_tmp[support_column] > 0.5).mean()
            results[f'Type I error > {80}%'] = type_1_error

            # Proportion of times support was bigger than 70% among splits in the true tree
            print((df[df[truth_column] == 1]).shape)
            print((df[df[truth_column] == 1][support_column] >= 0.5).shape)
            true_support = (df[df[truth_column] == 1][support_column] >= 0.5).mean()
            #print(true_support)
            results[f'Support > {80}% in True Tree'] = true_support
            return results

        else:
            #df['bound_dist_5'] = abs(df[support_column] - df['lower_5'])
            df_filtered_uncertainty = df[df["uncertainty"] <= bound]
            print("Fraction considered")
            print(df_filtered_uncertainty.shape[0]/df.shape[0])
            df_tmp = df_filtered_uncertainty[df_filtered_uncertainty[truth_column] == 0] # all not in true tree and below uncertainty
            not_in_true = df_tmp.shape[0]
            print("Not in true tree (after filtering):")
            print(not_in_true)
            print("In true tree (after filtering):")
            print(df_filtered_uncertainty[df_filtered_uncertainty[truth_column] == 1].shape[0])
            print("Not in true tree AND prediction >= 0.5:")
            print(df_tmp[df_tmp[support_column] >= 0.5].shape[0])




            #print(df_tmp[support_column])
            type_1_error = (df_tmp[support_column] >= 0.5).mean() # all not in true tree where EBG says >= 0.5

            results[f'Type I error > {80}%'] = type_1_error
            #print(f"Fraction EBG support >= 0.5 in filtered {(df_tmp[support_column] >= 0.5).shape}")
            #print(f"Fraction EBG support < 0.5 in filtered {(df_tmp[support_column] < 0.5).shape}")

            print(f"type 1 {type_1_error}")

            # Proportion of times support was bigger than 70% among splits in the true tree
            true_support = (df_filtered_uncertainty[df_filtered_uncertainty[truth_column] == 1][support_column] >= 0.5).mean()
            results[f'Support > {80}% in True Tree'] = true_support
            print(f"type 2 {true_support}")
            print("-"*20)

            return results
    # Proportion of times support was bigger than 70%, 80%, and 90% among splits not in the true tree (Type I error)
    for threshold in [80]:
        if not filtered:
            df_tmp = df[df[truth_column] == 0]
            type_1_error = (df_tmp[support_column] > threshold).mean()
            results[f'Type I error > {threshold}%'] = type_1_error

            # Proportion of times support was bigger than 70% among splits in the true tree
            true_support = (df[df[truth_column] == 1][support_column] > threshold).mean()
            results[f'Support > {threshold}% in True Tree'] = true_support

        else:
            df['bound_dist_5'] = abs(df[support_column] - df['lower_5'])
            df_filtered_uncertainty = df[df["bound_dist_5"] <= bound]
            print(df_filtered_uncertainty.shape[0]/df.shape[0])

            df_tmp = df_filtered_uncertainty[df_filtered_uncertainty[truth_column] == 0]
            type_1_error = (df_tmp[support_column] > threshold).mean()
            results[f'Type I error > {threshold}%'] = type_1_error

            # Proportion of times support was bigger than 70% among splits in the true tree
            true_support = (df_filtered_uncertainty[df_filtered_uncertainty[truth_column] == 1][support_column] > threshold).mean()
            results[f'Support > {threshold}% in True Tree'] = true_support

    return results


# Calculate the measures for each dataset
measures_ebg = calculate_measures(df_ebg, 'EBG_support_prob_80', 'in_true', ebg=True)
measures_ebg_filtered_09= calculate_measures(df_ebg_uncertainty.copy(), support_column='EBG_support_prob_80', truth_column='in_true', bound=0.8, filtered=True, ebg=True)
measures_ebg_filtered_06 = calculate_measures(df_ebg_uncertainty.copy(), support_column='EBG_support_prob_80', truth_column='in_true', bound=0.5, filtered=True, ebg=True)
measures_ebg_filtered_03 = calculate_measures(df_ebg_uncertainty.copy(), support_column='EBG_support_prob_80', truth_column='in_true', bound=0.3, filtered=True, ebg=True)
measures_ebg_filtered_01 = calculate_measures(df_ebg_uncertainty.copy(), support_column='EBG_support_prob_80', truth_column='in_true', bound=0.17, filtered=True, ebg=True)
measures_ebg_filtered_005 = calculate_measures(df_ebg_uncertainty.copy(), support_column='EBG_support_prob_80', truth_column='in_true', bound=0.16, filtered=True, ebg=True)

measures_alrt = calculate_measures(df_alrt, 'alrt_Support', 'inTrue')
measures_ufb = calculate_measures(df_ufb, 'ufb_Support', 'inTrue')
measures_sbs = calculate_measures(df_sbs, 'sbs_Support', 'inTrue')
measures_rb = calculate_measures(df_rb, 'rb_Support', 'inTrue')

sys.exit()
df_alrt_un = df_alrt.merge(df_ebg_uncertainty, on=["dataset","branchID_True"], how="inner")
print(df_alrt_un.shape)
sys.exit()
df_ufb_un = df_ufb.merge(df_ebg_uncertainty, on=["dataset","branchID_True"], how="inner")
df_sbs_un = df_sbs.merge(df_ebg_uncertainty, on=["dataset","branchID_True"], how="inner")
df_rb_un = df_rb.merge(df_ebg_uncertainty, on=["dataset","branchID_True"], how="inner")

measures_alrt_filtered = calculate_measures(df_alrt_un, 'alrt_Support', 'inTrue', filtered=True, ebg=True, bound=0.3)
measures_ufb_filtered = calculate_measures(df_ufb_un, 'ufb_Support', 'inTrue', filtered=True, ebg=True, bound=0.3)
measures_sbs_filtered = calculate_measures(df_sbs_un, 'sbs_Support', 'inTrue', filtered=True, ebg=True, bound=0.3)
measures_rb_filtered = calculate_measures(df_rb_un, 'rb_Support', 'inTrue', filtered=True, ebg=True, bound=0.3)

# Combine the results into a DataFrame
results_df = pd.DataFrame({
    'EBG': measures_ebg,
    'EBG filtered (80)': measures_ebg_filtered_09,
    'EBG filtered (50)': measures_ebg_filtered_06,
    'EBG filtered (30)': measures_ebg_filtered_03,
    'EBG filtered (17)': measures_ebg_filtered_01,
    'EBG filtered (16)': measures_ebg_filtered_005,
    'ALRT': measures_alrt,
    'UFB': measures_ufb,
    'SBS': measures_sbs,
    'RB': measures_rb,
    'ALRT_filtered': measures_alrt_filtered,
    'UFB_filtered': measures_ufb_filtered,
    'SBS_filtered': measures_sbs_filtered,
    'RB_filtered': measures_rb_filtered,

})

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# Transpose the DataFrame for better readability
results_df = results_df.T
results_df = results_df.round(2)

# Print the results in tabular format
results_df.to_csv(os.path.join(os.pardir, "data/comparison/simulations/alternative_measures_new.csv"), index=True)
print(results_df)

fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Bar plot for Type I errors
type_i_errors = results_df[[f'Type I error > {threshold}%' for threshold in [70, 75, 80, 85]]]
type_i_errors.plot(kind='bar', ax=ax[0])
ax[0].set_ylabel('Proportion')
ax[0].set_xlabel('Datasets')
ax[0].legend(title='Threshold')

# Bar plot for Support > 70% in True Tree
type_i_errors = results_df[[f'Support > {threshold}% in True Tree' for threshold in [70, 75, 80, 85]]]
type_i_errors.plot(kind='bar', ax=ax[1], color='green')
ax[1].set_ylabel('Proportion')
ax[1].set_xlabel('Datasets')

plt.tight_layout()
plt.show()

import pandas as pd

# Example data
data = {
    'Method': ['EBG', 'EBG filtered (D: 25)', 'EBG filtered (D: 15)', 'EBG filtered (D: 10)', 'EBG filtered (D: 5)',
               'EBG filtered (D: 1)', 'ALRT', 'UFB', 'SBS', 'RB'],
    'Fraction considered': [1.0, 0.92, 0.65, 0.52, 0.41, 0.27, 1.0, 1.0, 1.0, 1.0],
    'FP fraction (>70)': [0.15, 0.14, 0.07, 0.05, 0.03, 0.01, 0.23, 0.27, 0.18, 0.18],
    'TP fraction (>70)': [0.71, 0.72, 0.8, 0.86, 0.89, 0.9, 0.85, 0.9, 0.8, 0.79],
    'FP fraction (>80)': [0.12, 0.11, 0.07, 0.05, 0.03, 0.01, 0.17, 0.23, 0.15, 0.15],
    'TP fraction (>80)': [0.62, 0.65, 0.78, 0.86, 0.89, 0.9, 0.74, 0.85, 0.72, 0.72],
    'FP fraction (>90)': [0.08, 0.08, 0.07, 0.05, 0.03, 0.01, 0.11, 0.17, 0.12, 0.11],
    'TP fraction (>90)': [0.49, 0.54, 0.76, 0.86, 0.89, 0.9, 0.55, 0.76, 0.61, 0.61]
}

df = pd.DataFrame(data)


def calculate_f1(tp, fp):
    if tp + fp == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp  # Assuming recall is given as tp directly
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# Calculate F1 scores for each threshold
thresholds = [70, 80, 90]
f1_scores = {f'F1 > {t}%': [] for t in thresholds}

for t in thresholds:
    fp_col = f'FP fraction (>{t})'
    tp_col = f'TP fraction (>{t})'

    for index, row in df.iterrows():
        fp = row[fp_col]
        tp = row[tp_col]
        f1_score = calculate_f1(tp, fp)
        f1_scores[f'F1 > {t}%'].append(f1_score)

# Add F1 scores to the DataFrame
for t in thresholds:
    df[f'F1 > {t}%'] = f1_scores[f'F1 > {t}%']

# Print the DataFrame
print(df[['Method', 'F1 > 70%', 'F1 > 80%', 'F1 > 90%']])
df = df.round(2)
df.to_csv(os.path.join(os.pardir, "data/comparison/simulations/alternative_measures_f1.csv"), index=False)

