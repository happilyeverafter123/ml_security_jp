import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

df = pd.read_csv(
    os.environ.get("TRAIN_DATA_PATH")
)



def H_entropy(x):
    prob = [ float(x.count(c)) / len(x) for c in dict.fromkeys(list(x)) ]
    H = - sum([p * np.log2(p) for p in prob])
    return H

def func_preprocessing(df):
    train_rows = ((df.attack_type == 'norm') | (df.attack_type == 'sqli'))
    df = df[train_rows]
    
    entropies = []
    closing_parentheses = []
    
    for i in df['payload']:
        entropies.append(H_entropy(i))
        if i.count(')') > 0:
            closing_parentheses.append(i.count(')'))
        else:
            closing_parentheses.append(0)
            
    df = df.assign(entropy=entropies)
    df = df.assign(closing_parenthesis=closing_parentheses)
    
    rep = df.label.replace({'norm': 0, 'anom': 1})
    df = df.assign(label=rep)
    
    return df

df = func_preprocessing(df)

test_data = pd.read_csv(
    os.environ.get("TEST_DATA_PATH")
)

test_data = func_preprocessing(test_data)

# df_x = df[['length', 'entropy', 'closing_parenthesis']]
# test_data_x = test_data[['length', 'entropy', 'closing_parenthesis']]
# df_y = df[['label']]
# test_data_y = test_data[['label']]

# X_all = pd.concat([df_x, test_data_x])
# y_all = pd.concat([df_y, test_data_y])

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# import numpy as np
# import optuna
# from sklearn.model_selection import cross_val_score

# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=True, random_state=42)

# class Objective_DTC:
#     def __init__(self):
#         self.X = X
#         self.y = y
#         
#     def __call__(self, trial):
#         params = {
#             "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
#             "max_depth": trial.suggest_int("max_depth", 1, 10),
#             "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
#             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
#         }
#         
#         model = DecisionTreeClassifier(**params)
#         score = cross_val_score(model, self.X, self.y.values.ravel(), n_jobs=-1, cv=3)
#         return score.mean()
    
# objective = Objective_DTC(X_train, y_train)
# study = optuna.create_study()
# study.optimize(objective, timeout=600)
# best_params = study.best_params

# train_rows = ((df.attack_type == 'norm') | (df.attack_type == 'sqli'))
# df = df[train_rows]

# test_train_rows = ((test_data.attack_type == 'norm') | (test_data.attack_type == 'sqli'))
# test_data = test_data[test_train_rows]

df_y = df[['label']]
test_y = test_data[['label']]

df_x = df.drop(columns=['label'])
test_x = test_data.drop(columns=['label'])

X_all = pd.concat([df_x, test_x])
y_all = pd.concat([df_y, test_y])

rep = y_all.label.replace({'norm': 0, 'anom': 1})
y_all = y_all.assign(label=rep)

# print(X_all.head())

X = X_all['payload']
y = y_all

vec_opts = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "max_features": 10000,
    "analyzer": "char",
    "min_df": 0.1
}

v = TfidfVectorizer(**vec_opts)

X = v.fit_transform(X)

feature_names = v.get_feature_names_out()


simple_feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# DataFrame を作成
df = pd.DataFrame(X.toarray(), columns=simple_feature_names)
# print(np.array(feature_names))

print(df.head(30))
print(y.head(30))

# save_path = os.environ.get("SAVE_PATH")
# df.to_csv(save_path, index=False)
# print(df.head(30))
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import optuna
import optuna.integration.lightgbm as olgb
import lightgbm as lgb

X_train, X_test, y_train, y_test = train_test_split(df, y.values.ravel(), test_size=0.2, shuffle=True, random_state=42)

# LightGBM用のデータセットに変換
lgb_train = olgb.Dataset(X_train, y_train)

# パラメータの設定
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
}

tuner = olgb.LightGBMTunerCV(
    params,
    lgb_train,
    num_boost_round=1000
)

tuner.run()
# # 交差検証を使用したハイパーパラメータの探索
# tuner = olgb.LightGBMTuner(params, train, valid_sets=[test])

best_params = tuner.best_params
best_score = tuner.best_score
print("Best score found by optuna is:", best_score)
print("Best parameters found by optuna are:", best_params)
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))

lgb_test = olgb.Dataset(X_test, y_test)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'lambda_l1': best_params['lambda_l1'],
    'lambda_l2': best_params['lambda_l2'],
    'num_leaves': best_params['num_leaves'],
    'feature_fraction': best_params['feature_fraction'],
    'bagging_fraction': best_params['bagging_fraction'],
    'bagging_freq': best_params['bagging_freq'],
    'min_child_samples': best_params['min_child_samples']
}

gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    verbose_eval=0
)

lgb_preds = gbm.predict(X_test)
lgb_preds = (lgb_preds > 0.5).astype(int)

import xgboost as xgb

def objective(trial):
    eta = trial.suggest_loguniform("eta", 1e-8, 1.0)
    gamma = trial.suggest_loguniform("gamma", 1e-8, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 10)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    max_delta_step = trial.suggest_int("max_delta_step", 0, 10)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 1.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 1.0)
    
    regr = xgb.XGBRegressor(
        eta =eta,
        gamma =gamma,
        max_depth =max_depth,
        min_child_weight =min_child_weight,
        max_delta_step =max_delta_step,
        subsample =subsample,
        reg_lambda =reg_lambda,
        reg_alpha =reg_alpha,
    )
    
    regr.fit(X_train, y_train)
    
    pred = regr.predict(X_test)
    pred = (pred > 0.5).astype(int)
    pred_labels = np.rint(pred)
    
    accuracy = accuracy_score(y_test, pred_labels)
    return (1 - accuracy)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

optimized_model = xgb.XGBRegressor(
    eta = study.best_params['eta'],
    gamma = study.best_params['gamma'],
    max_depth = study.best_params['max_depth'],
    min_child_weight = study.best_params['min_child_weight'],
    max_delta_step = study.best_params['max_delta_step'],
    subsample = study.best_params['subsample'],
    reg_lambda = study.best_params['reg_lambda'],
    reg_alpha = study.best_params['reg_alpha']
)

optimized_model.fit(X_train, y_train)

xgb_preds = optimized_model.predict(X_test)
xgb_preds = (xgb_preds > 0.5).astype(int)

combined_preds = (lgb_preds * 0.5 + xgb_preds * 0.5).round().astype(int)

combined_accuracy = accuracy_score(y_test, combined_preds)
print("Combined model accuracy:", combined_accuracy)

conf_matrix = confusion_matrix(y_test, combined_preds)
print("Confusion Matrix:")
# pred_labels_xgb = optimized_model.predict(X_test)
# pred_labels_xgb = (pred_labels_xgb > 0.5).astype(int)

# pred_labels_xgb_rounded = np.rint(lgb_preds)
