import xgboost as xgb
from sklearn.datasets import load_boston

boston = load_boston()

# XGBoost API example
params = {'tree_method': 'gpu_hist', 'max_depth': 3, 'learning_rate': 0.1, 'gpu_id': 7}
dtrain = xgb.DMatrix(boston.data, boston.target)
xgb.train(params, dtrain, evals=[(dtrain, "train")])

# sklearn API example
gbm = xgb.XGBRegressor(silent=False, n_estimators=10, tree_method='gpu_hist', gpu_id=7)
gbm.fit(boston.data, boston.target, eval_set=[(boston.data, boston.target)]) 