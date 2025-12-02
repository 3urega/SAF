import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import torch.nn as nn
import torch
import pandas as pd
import numpy as np


def get_XGBoost(X, Y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',  # regression
        'learning_rate': 0.05,
        'max_depth': 4,
        'monotone_constraints': '(0,0,0,-1,0,0,0,-1)',
        'eval_metric': 'rmse',
    }

    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    bst = xgb.train(params,
                    dtrain,
                    num_boost_round=400,
                    evals=evallist,
                    early_stopping_rounds=20,
                    verbose_eval=True)
    
    y_pred = bst.predict(dtest)
    print(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)
    
    return bst

def get_GradientBoost(X, y, n_estimators=200, max_depth = 3):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    return model