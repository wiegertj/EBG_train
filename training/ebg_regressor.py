import math
import random
import lightgbm as lgb
import os
import pickle
import optuna
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import GroupKFold


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


def light_gbm_regressor(rfe=False, rfe_feature_n=20):
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
    rmse_baseline = mean_squared_error(y_test, X_test["parsimony_bootstrap_support"], squared=False)
    mbe_baseline = MBE(y_test, X_test["parsimony_bootstrap_support"])
    mae_baseline = mean_absolute_error(y_test, X_test["parsimony_bootstrap_support"])
    mdae_baseline = median_absolute_error(y_test, X_test["parsimony_bootstrap_support"])

    print("MSE (Mean Squared Error):", mse_baseline)
    print("RMSE (Root Mean Squared Error):", rmse_baseline)
    print("MBE :", mbe_baseline)
    print("MAE (Mean Absolute Error):", mae_baseline)
    print("MdAE (Median Absolute Error):", mdae_baseline)

    if rfe:
        model = RandomForestRegressor(n_jobs=-1, n_estimators=250, max_depth=10, min_samples_split=20,
                                      min_samples_leaf=10)
        rfe = RFE(estimator=model, n_features_to_select=rfe_feature_n)  # Adjust the number of features as needed
        rfe.fit(X_train.drop(axis=1, columns=['dataset', 'branchId', 'group']), y_train)
        print(rfe.support_)
        selected_features = X_train.drop(axis=1, columns=['dataset', 'branchId', 'group']).columns[rfe.support_]
        selected_features = selected_features.append(pd.Index(['group']))

        print("Selected features for RFE: ")
        print(selected_features)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    X_test_ = X_test
    if not rfe:
        X_train = X_train.drop(axis=1, columns=['dataset'])
        X_test = X_test.drop(axis=1, columns=['dataset'])

    val_scores_median = []

    def objective_median(trial):

        params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'alpha': 0.5,
            'num_iterations': trial.suggest_int('num_iterations', 50, 300),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 200),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.5),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 1e-5, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 1e-5, 1.0),
            'min_split_gain': trial.suggest_uniform('min_split_gain', 1e-5, 0.3),
            'bagging_freq': 0,
            "verbosity": -1
        }

        val_scores = []

        gkf = GroupKFold(n_splits=6)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            model = lgb.train(params, train_data)
            val_preds = model.predict(X_val)
            val_score = quantile_loss(y_val, val_preds, 0.5)
            print("score: " + str(val_score))

            val_scores.append(val_score)
            val_scores_median.append(val_score)

        return sum(val_scores) / len(val_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_median, n_trials=5)
    df = pd.DataFrame({'Value': val_scores_median})

    df.to_csv('val_scores.csv', index=False)
    best_params = study.best_params
    best_params["objective"] = "quantile"
    best_params["metric"] = "quantile"
    best_params["boosting_type"] = "gbdt"
    best_params["bagging_freq"] = 0
    best_params["alpha"] = 0.5
    best_params["verbosity"] = -1
    best_score_median = study.best_value

    print(f"Best Params: {best_params}")
    print(f"Best MAPE training: {best_score_median}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)

    final_model = lgb.train(best_params, train_data)

    #model_path = os.path.join(os.pardir, "data/processed/final", "test_median_model.pkl")
    #with open(model_path, 'wb') as file:
    #    pickle.dump(final_model, file)

    y_pred_median = final_model.predict(X_test.drop(axis=1, columns=["group"]))

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

    def objective_lower_bound_5(trial):

        params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'alpha': 0.05,
            'num_iterations': trial.suggest_int('num_iterations', 10, 300),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 200),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.3),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 1e-5, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 1e-5, 1.0),
            'min_split_gain': trial.suggest_uniform('min_split_gain', 1e-5, 0.3),
            'bagging_freq': 0,
            "verbosity": -1
        }

        val_scores = []

        gkf = GroupKFold(n_splits=5)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]
            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            model = lgb.train(params, train_data)
            val_preds = model.predict(X_val)
            val_score = quantile_loss(y_val, val_preds, 0.05)
            val_scores.append(val_score)

        return sum(val_scores) / len(val_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_lower_bound_5, n_trials=5)

    best_params_lower_bound = study.best_params
    best_params_lower_bound["objective"] = "quantile"
    best_params_lower_bound["metric"] = "quantile"
    best_params_lower_bound["boosting_type"] = "gbdt"
    best_params_lower_bound["bagging_freq"] = 0
    best_params_lower_bound["alpha"] = 0.05
    best_params_lower_bound["verbosity"] = -1
    best_score_lower_bound = study.best_value

    print(f"Best Params: {best_params_lower_bound}")
    print(f"Best Quantile Loss: {best_score_lower_bound}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)

    final_model_lower_bound_5 = lgb.train(best_params_lower_bound, train_data)

    model_path = os.path.join(os.pardir, "data/processed/final", "test_low_model_5.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(final_model_lower_bound_5, file)

    y_pred_lower_5 = final_model_lower_bound_5.predict(X_test.drop(axis=1, columns=["group"]))
    print("Quantile Loss on Holdout: " + str(quantile_loss(y_test, y_pred_lower_5, 0.05)))

    def objective_lower_bound_10(trial):

        params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'alpha': 0.1,
            'num_iterations': trial.suggest_int('num_iterations', 10, 300),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 200),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.3),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 1e-5, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 1e-5, 1.0),
            'min_split_gain': trial.suggest_uniform('min_split_gain', 1e-5, 0.3),
            'bagging_freq': 0,
            "verbosity": -1
        }

        val_scores = []

        gkf = GroupKFold(n_splits=5)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]
            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            model = lgb.train(params, train_data)
            val_preds = model.predict(X_val)
            val_score = quantile_loss(y_val, val_preds, 0.1)
            val_scores.append(val_score)

        return sum(val_scores) / len(val_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_lower_bound_10, n_trials=5)

    best_params_lower_bound = study.best_params
    best_params_lower_bound["objective"] = "quantile"
    best_params_lower_bound["metric"] = "quantile"
    best_params_lower_bound["boosting_type"] = "gbdt"
    best_params_lower_bound["bagging_freq"] = 0
    best_params_lower_bound["alpha"] = 0.1
    best_params_lower_bound["verbosity"] = -1
    best_score_lower_bound = study.best_value

    print(f"Best Params: {best_params_lower_bound}")
    print(f"Best Quantile Loss: {best_score_lower_bound}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)

    final_model_lower_bound_10 = lgb.train(best_params_lower_bound, train_data)

    #model_path = os.path.join(os.pardir, "data/processed/final", "test_low_model_10.pkl")
    #with open(model_path, 'wb') as file:
     #   pickle.dump(final_model_lower_bound_10, file)

    y_pred_lower_10 = final_model_lower_bound_10.predict(X_test.drop(axis=1, columns=["group"]))
    print("Quantile Loss on Holdout: " + str(quantile_loss(y_test, y_pred_lower_10, 0.1)))

    X_test_["prediction_median"] = y_pred_median
    X_test_["prediction_low_5"] = y_pred_lower_5
    X_test_["prediction_low_10"] = y_pred_lower_10
    X_test_["support"] = y_test
    X_test_["pred_error"] = y_test - y_pred_median

    #X_test_.to_csv(os.path.join(os.pardir, "data/processed/final", "test_regressor.csv"))


light_gbm_regressor(rfe=False)
