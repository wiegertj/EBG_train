import pickle
import random
import lightgbm as lgb
import os
import optuna
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import entropy


def light_gbm_classifier(threshold, rfe=False, rfe_feature_n=10, train_light=True):
    df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final.csv"))

    if train_light:
        group1 = ["mean_closeness_centrality_ratio", "branch_length_ratio_split"]
        group2 = ['min_pars_support_children_weighted', 'max_pars_support_children_weighted',
                  'mean_pars_support_parents_weighted', 'min_pars_support_children', 'std_pars_support_children',
                  'number_children_relative', 'mean_pars_support_children_weighted',
                  'mean_pars_bootstrap_support_parents', 'std_pars_bootstrap_support_parents',
                  'min_pars_bootstrap_support_children_w', 'max_pars_bootstrap_support_children_w',
                  'std_pars_bootstrap_support_children']
        group3 = ['max_substitution_frequency', 'mean_substitution_frequency', 'cv_substitution_frequency',
                  'skw_substitution_frequency', 'Unnamed: 0']

        group_to_drop = group1 + group2 + group3
        df = df.drop(columns=group_to_drop, errors='ignore')
    """
    This functions trains the classifier to solve the binary classification problem between the class 0 
    (SBS value does not exceed threshold) and the class 1 (it does). 
    The final classifier will be stored to data/processed/final.

    Parameters
    ----------
    threshold : list of float
        thresholds for the SBS values

    rfe : bool
        wether to perform recursive feature elimination or not
    rfe_feature : int
        number of features for recursive feature elimination 
    """
    df.fillna(-1, inplace=True)
    df.replace([np.inf, -np.inf], -1, inplace=True)


    """
        Load regressions models and make prediction for using them as features for the classifer
    """
    df_reg_pred = df.drop(columns=["dataset", "support"], axis=1)
    print(df_reg_pred.columns)
    with open(os.path.join(os.path.pardir, "data", "processed", "final", "test_median_model_light.pkl"),
              'rb') as model_file:
        regression_median = pickle.load(model_file)

    with open(os.path.join(os.path.pardir, "data", "processed", "final", "test_low_model_10_light.pkl"),
              'rb') as model_file:
        regression_lower10 = pickle.load(model_file)

    with open(os.path.join(os.path.pardir, "data", "processed", "final", "test_low_model_5_light.pkl"),
              'rb') as model_file:
        regression_lower5 = pickle.load(model_file)

    df["median_pred"] = regression_median.predict(df_reg_pred)
    df["lower_bound_10"] = regression_lower10.predict(df_reg_pred)
    df["lower_bound_5"] = regression_lower5.predict(df_reg_pred)

    df.columns = df.columns.str.replace(':', '_')
    df["is_valid"] = 0
    df.loc[df['support'] > threshold, 'is_valid'] = 1

    """
        Filter 20% of complete datasets as holdout
    """
    df["group"] = df['dataset'].astype('category').cat.codes.tolist()
    target = "is_valid"
    df.drop(columns=["support"], inplace=True, axis=1)
    sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
    test = df[df['group'].isin(sample_dfs)]
    train = df[~df['group'].isin(sample_dfs)]

    X_train = train.drop(axis=1, columns=target)
    y_train = train[target]

    X_test = test.drop(axis=1, columns=target)
    y_test = test[target]

    X_train.fillna(-1, inplace=True)
    X_train.replace([np.inf, -np.inf], -1, inplace=True)

    if rfe:
        model = RandomForestRegressor(n_jobs=-1, n_estimators=250, max_depth=20, min_samples_split=20,
                                      min_samples_leaf=10)
        rfe = RFE(estimator=model, n_features_to_select=rfe_feature_n,
                  step=0.1)
        rfe.fit(X_train.drop(axis=1, columns=['dataset', 'group']), y_train)
        print(rfe.support_)
        selected_features = X_train.drop(axis=1, columns=['dataset', 'group']).columns[rfe.support_]
        selected_features = selected_features.append(pd.Index(['group']))

        print("Selected features for RFE: ")
        print(selected_features)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    X_test_ = X_test
    if not rfe:
        X_train = X_train.drop(axis=1, columns=['dataset'])
        X_test = X_test.drop(axis=1, columns=['dataset'])

    """
        Train classifier using optuna and GroupKFold Cross Validation, make holdout prediction and evaluate
    """
    def objective(trial):

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_iterations': trial.suggest_int('num_iterations', 10, 300),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 300),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.3),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 1e-5, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 1e-5, 1.0),
            'min_split_gain': trial.suggest_uniform('min_split_gain', 1e-5, 0.3),
            'bagging_freq': 0,
            'verbosity': -1
        }

        val_scores = []

        gkf = GroupKFold(n_splits=10)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            model = lgb.train(params, train_data)
            val_preds = model.predict(X_val)
            val_preds_binary = (val_preds >= 0.5).astype(int)

            val_score = f1_score(y_val, val_preds_binary)
            val_scores.append(val_score)

        return np.mean(val_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)

    best_params = study.best_params
    best_params["objective"] = "binary"
    best_params["metric"] = "binary_logloss"
    best_params["bagging_freq"] = 0
    best_score = study.best_value

    print(f"Best Params: {best_params}")
    print(f"Best F1 training: {best_score}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)
    final_model = lgb.train(best_params, train_data)

    model_path = os.path.join(os.pardir, "data/processed/final", f"test_classifier_{str(threshold)}_light.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(final_model, file)

    y_pred = final_model.predict(X_test.drop(axis=1, columns=["group"]))

    y_pred_binary = (y_pred >= 0.5).astype(int)
    used_probability = np.where(y_pred_binary == 1, y_pred, 1 - y_pred)
    not_probability = 1 - used_probability
    X_test_["used_probability"] = used_probability
    X_test_["not_probability"] = not_probability
    X_test_["entropy"] = -1

    for index, row in X_test_.iterrows():
        entropy_row = entropy([row["used_probability"], row["not_probability"]], base=2)  # Compute Shannon entropy
        X_test_.loc[index, 'entropy'] = entropy_row

    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')

    feature_importance = final_model.feature_importance(importance_type='gain')

    importance_df = pd.DataFrame(
        {'Feature': X_train.drop(axis=1, columns=["group"]).columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    scaler = MinMaxScaler()
    importance_df['Importance'] = scaler.fit_transform(importance_df[['Importance']])
    importance_df = importance_df.nlargest(30, 'Importance')

    print("Feature Importances (Normalized):")
    for index, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    X_test_["prediction"] = y_pred
    X_test_["prediction_binary"] = y_pred_binary
    X_test_["support"] = y_test
    X_test_.to_csv(os.path.join(os.pardir, "data/processed/final/", f"test_classifier_{threshold}_light" + ".csv"))


for threshold in [0.7, 0.75, 0.8, 0.85]:
    light_gbm_classifier(threshold, rfe=False, rfe_feature_n=10)
