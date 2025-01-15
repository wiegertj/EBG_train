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
    sample_indices = np.random.choice(len(X_train), size=500, replace=False)

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Randomly sample 10,000 indices
    sample_indices = np.random.choice(len(X_train_np), size=500, replace=False)

    # Create the sampled training data
    X_sample = X_train_np[sample_indices]
    y_sample = y_train_np[sample_indices]

    # Train the classifier on the sampled data
    clf.fit(X_sample, y_sample)

    y_pred_median = clf.predict(X_test.to_numpy())

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantile_predictions = clf.predict(
        X_test,
        output_type="quantiles",
        quantiles=quantiles,
    )

    print(quantile_predictions.shape)

    mse = mean_squared_error(y_test, y_pred_median)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    r_squared = r2_score(y_test, y_pred_median)
    print(f"R-squared on test set: {r_squared:.2f}")

    mae = mean_absolute_error(y_test, y_pred_median)
    print(f"MAE on test set: {mae:.2f}")

    mbe = MBE(y_test, y_pred_median)
    print(f"MBE on test set: {mbe:.2f}")

    mdae = median_absolute_error(y_test, y_pred_median)
    print(f"MdAE on test set: {mdae}")

    # Assuming quantile_predictions and y_test are already available
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    # Split predictions into individual quantiles
    q05, q25, q50, q75, q95 = [quantile_predictions[:, i] for i in range(len(quantiles))]

    # Calculate narrowness of the quantile region
    quantile_widths = {
        'Q95-Q05': q95 - q05,
        'Q75-Q25': q75 - q25
    }

    # Initialize results storage
    results = {
        'Region': [],
        'Width_Mean': [],
        'MAE': [],
        'MdAE': []
    }

    # Calculate errors for each region
    for region, width in quantile_widths.items():
        # Use the median prediction (q50) as the central value
        mae = mean_squared_error(y_test, q50, squared=False)  # RMSE as MAE proxy
        mdae = median_absolute_error(y_test, q50)

        results['Region'].append(region)
        results['Width_Mean'].append(np.mean(width))
        results['MAE'].append(mae)
        results['MdAE'].append(mdae)

    # Convert results to a structured display
    results_df = pd.DataFrame(results)

    # Display results to the user
    # Plot narrowness vs errors
    plt.figure(figsize=(10, 6))
    plt.plot(results['Width_Mean'], results['MAE'], marker='o', label='MAE')
    plt.plot(results['Width_Mean'], results['MdAE'], marker='o', label='MdAE')
    plt.xlabel('Mean Width of Quantile Region')
    plt.ylabel('Error Metrics')
    plt.title('Impact of Quantile Region Narrowness on Error Metrics')
    plt.legend()
    plt.grid()
    plt.savefig("/hits/fast/cme/wiegerjs/tabpfn.png")


tabfn_regressor()
