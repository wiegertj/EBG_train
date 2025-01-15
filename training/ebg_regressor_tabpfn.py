import math
import random
import os
import pickle
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt


def quantile_loss(y_true, y_pred, quantile):
    """
    Computes the quantile loss.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values

    y_pred : np.ndarray
        Prediction values

    quantile : float in [0,1]
        Quantile, e.g. 0.5 for the median quantile
    """
    residual = y_true - y_pred
    return mean(np.maximum(quantile * residual, (quantile - 1) * residual))


def MBE(y_true, y_pred):
    """
    Computes Mean Bias Error (MBE)

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values

    y_pred : np.ndarray
        Prediction values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_true - y_pred)
    mbe = diff.mean()
    return mbe


def tabfn_regressor():
    """
    Function for training the three EBG regressors (5%/10% lower bound and median prediction).
    Reads in the final_dataset at data/processed/final and then performs quantile regression three times.
    Each time 100 optuna trials are used for hyperparameter tuning.
    The three resulting models are stored at data/processed/final.

    Parameters
    ----------
    rfe : bool
        wether to perform recursive feature elimination or not

    rfe_feature_n : int
        number of features for recursive feature elimination
    """
    df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final.csv"),
                     usecols=lambda column: column != 'Unnamed: 0')

    # drop features for EBG-light training
    group2 = [
        'min_pars_support_children_weighted',
        'max_pars_support_children_weighted',
        'mean_pars_support_parents_weighted',
        'min_pars_support_children',
        'std_pars_support_children',
        'number_children_relative',
        'mean_pars_support_children_weighted',
        'mean_pars_bootstrap_support_parents',
        'std_pars_bootstrap_support_parents',
        'min_pars_bootstrap_support_children_w',
        'max_pars_bootstrap_support_children_w',
        'std_pars_bootstrap_support_children'
    ]
    df = df.drop(columns=group2, errors='ignore')

    df.fillna(-1, inplace=True)
    df.replace([np.inf, -np.inf], -1, inplace=True)

    print("Median Support: ")
    print(df["support"].median())
    df.columns = df.columns.str.replace(':', '_')
    df["group"] = df['dataset'].astype('category').cat.codes.tolist()

    target = "support"
    sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
    test = df[df['group'].isin(sample_dfs)]
    train = df[~df['group'].isin(sample_dfs)]

    X_train = train.drop(axis=1, columns=target)
    y_train = train[target]

    X_test = test.drop(axis=1, columns=target)
    y_test = test[target]

    mse_zero = mean_squared_error(y_test, np.zeros(len(y_test)))
    rmse_zero = math.sqrt(mse_zero)
    print("Baseline prediting 0 RMSE: " + str(rmse_zero))

    mse_mean = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train))
    rmse_mean = math.sqrt(mse_mean)
    print("Baseline predicting mean RMSE: " + str(rmse_mean))

    mse_baseline = mean_squared_error(y_test, X_test["parsimony_bootstrap_support"])
    rmse_baseline = mean_squared_error(y_test, X_test["parsimony_bootstrap_support"])
    mbe_baseline = MBE(y_test, X_test["parsimony_bootstrap_support"])
    mae_baseline = mean_absolute_error(y_test, X_test["parsimony_bootstrap_support"])
    mdae_baseline = median_absolute_error(y_test, X_test["parsimony_bootstrap_support"])

    print("MSE (Mean Squared Error):", mse_baseline)
    print("RMSE (Root Mean Squared Error):", rmse_baseline)
    print("MBE :", mbe_baseline)
    print("MAE (Mean Absolute Error):", mae_baseline)
    print("MdAE (Median Absolute Error):", mdae_baseline)

    clf = TabPFNRegressor()
    sample_indices = np.random.choice(len(X_train), size=10000, replace=False)

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Randomly sample 10,000 indices
    sample_indices = np.random.choice(len(X_train_np), size=10000, replace=False)

    # Create the sampled training data
    X_sample = X_train_np[sample_indices]
    y_sample = y_train_np[sample_indices]

    # Train the classifier on the sampled data
    clf.fit(X_sample, y_sample)

    y_pred_median = clf.predict(X_test.to_numpy())

    # Assuming quantile_predictions and y_test are already available
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    quantile_predictions = clf.predict(
        X_test,
        output_type="quantiles",
        quantiles=quantiles,
    )



    print(quantile_predictions)
    print(type(quantile_predictions))

    # Split predictions into individual quantiles from the list
    q05, q25, q50, q75, q95 = quantile_predictions

    quantile_data = {'Q05': q05, 'Q25': q25, 'Q75': q75, 'Q95': q95}
    for quantile_name, quantile_values in quantile_data.items():
        plt.figure(figsize=(10, 6))
        plt.hist(quantile_values, bins=30, edgecolor='k', alpha=0.7)
        plt.xlabel(f'{quantile_name} Values')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {quantile_name}')
        plt.grid()
        plt.savefig(f"/hits/fast/cme/wiegerjs/hist_{quantile_name}.png")

    # Calculate interval widths
    intervals = {
        'Q75_Q25': q75 - q25,
        'Q95_Q05': q95 - q05
    }

    # Initialize results storage
    results = {
        'Interval': [],
        'Width_Bin': [],
        'MAE': [],
        'MdAE': []
    }

    # Analyze narrowness impact on median prediction (q50)
    for interval_name, interval_widths in intervals.items():
        # Bin the interval widths
        bins = np.linspace(interval_widths.min(), interval_widths.max(), num=10)
        bin_indices = np.digitize(interval_widths, bins)

        # Plot histogram of the distribution of the data in the bins
        plt.figure(figsize=(10, 6))
        plt.hist(interval_widths, bins=bins, edgecolor='k', alpha=0.7, density=True)
        plt.xlabel(f'{interval_name} Width')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {interval_name} Widths')
        plt.grid()
        plt.savefig(f"/hits/fast/cme/wiegerjs/tabpfn_hist_{interval_name}")

        for bin_idx in range(1, len(bins)):
            bin_mask = bin_indices == bin_idx
            if np.any(bin_mask):
                bin_width_mean = interval_widths[bin_mask].mean()
                mae = mean_absolute_error(y_test[bin_mask], q50[bin_mask])
                mdae = median_absolute_error(y_test[bin_mask], q50[bin_mask])

                results['Interval'].append(interval_name)
                results['Width_Bin'].append(bin_width_mean)
                results['MAE'].append(mae)
                results['MdAE'].append(mdae)
    # Display results to the user
    results_df = pd.DataFrame(results)

    # Plot narrowness vs errors for each interval
    # Plot narrowness vs errors for each interval
    for interval_name in intervals.keys():
        interval_data = results_df[results_df['Interval'] == interval_name]
        plt.figure(figsize=(10, 6))
        plt.plot(interval_data['Width_Bin'], interval_data['MAE'], marker='o', label='MAE')
        plt.plot(interval_data['Width_Bin'], interval_data['MdAE'], marker='o', label='MdAE')
        plt.xlabel('Mean Width of ' + interval_name)
        plt.ylabel('Error Metrics')
        plt.title(f'Impact of {interval_name} Narrowness on Error Metrics')
        plt.legend()
        plt.grid()
        plt.savefig(f"/hits/fast/cme/wiegerjs/tabpfn_{interval_name}.png")


tabfn_regressor()
