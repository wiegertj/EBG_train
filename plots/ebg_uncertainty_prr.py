import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, roc_curve, auc

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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math

def accuracy_rejection_curve(uncertainty_scores, true_labels,prediction_labels ,num_thresholds=100):
    # Generate thresholds
    thresholds = np.linspace(0, 1, num_thresholds)

    # Initialize arrays to store accuracy and rejection ratio
    accuracies = []
    rejection_percentages = []

    # Calculate accuracy and rejection ratio for each threshold
    for threshold in thresholds:
        # Filter predictions based on uncertainty threshold
        predictions = uncertainty_scores < threshold

        # Calculate accuracy for filtered predictions
        accuracy = accuracy_score(true_labels[predictions], prediction_labels[predictions])
        if math.isnan(accuracy):
            continue
        accuracies.append(accuracy)

        # Calculate rejection percentage
        rejection_percentage = (np.sum(predictions == False) / len(predictions))
        rejection_percentages.append(rejection_percentage)

    return accuracies, rejection_percentages


def prediction_rejection_ratio(accuracies, rejection_percentages):
    # Calculate the area under the Accuracy Rejection Curve (AUC_AR)
    auc_ar = auc(rejection_percentages, accuracies)

    # Calculate the area under the Random Classifier Curve (0.5)
    auc_random = 0.5

    # Calculate the area between the AR curve and the random curve
    area_between_ar_random = auc_ar - auc_random

    # Calculate the area between the perfect classifier curve and the random curve
    area_between_perfect_random = 1.0 - auc_random

    # Compute the Prediction Rejection Ratio (PRR)
    prr = area_between_ar_random / area_between_perfect_random

    return prr


# Example usage


# Calculate accuracies and rejection percentages using accuracy rejection curve




uncertainty_scores = df_merged["entropy_ebg_over_85"]
true_labels = df_merged["support_over_85"]
prediction_labels = df_merged["ebg_over_85"]
accuracies, rejection_percentages = accuracy_rejection_curve(uncertainty_scores, true_labels, prediction_labels)
prr = prediction_rejection_ratio(accuracies, rejection_percentages)
print("Prediction Rejection Ratio (PRR) 0.85:", prr)

uncertainty_scores = df_merged["entropy_ebg_over_80"]
true_labels = df_merged["support_over_80"]
prediction_labels = df_merged["ebg_over_80"]
accuracies, rejection_percentages = accuracy_rejection_curve(uncertainty_scores, true_labels, prediction_labels)
prr = prediction_rejection_ratio(accuracies, rejection_percentages)
print("Prediction Rejection Ratio (PRR) 0.80:", prr)

uncertainty_scores = df_merged["entropy_ebg_over_75"]
true_labels = df_merged["support_over_75"]
prediction_labels = df_merged["ebg_over_75"]
accuracies, rejection_percentages = accuracy_rejection_curve(uncertainty_scores, true_labels, prediction_labels)
prr = prediction_rejection_ratio(accuracies, rejection_percentages)
print("Prediction Rejection Ratio (PRR) 0.75:", prr)

uncertainty_scores = df_merged["entropy_ebg_over_70"]
true_labels = df_merged["support_over_70"]
prediction_labels = df_merged["ebg_over_70"]
accuracies, rejection_percentages = accuracy_rejection_curve(uncertainty_scores, true_labels, prediction_labels)
prr = prediction_rejection_ratio(accuracies, rejection_percentages)
print("Prediction Rejection Ratio (PRR) 0.70:", prr)