import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

submission = pd.read_csv('sample_submission.csv')

# 인덱스 드랍
train = train.drop(['index'], axis=1)
test = test.drop(['index'], axis=1)

# 원핫 인코딩
train_one = pd.get_dummies(train)
test_one = pd.get_dummies(test)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, Lasso

train_X = train_one.drop('INVC_CONT',axis = 1)
train_Y = train_one['INVC_CONT']

#모델 정의
# v1
# model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=400,
#                       subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0,
#                       min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0,
#                       colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
#                       importance_type='split')

# v2
# model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=400,
#                       subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
#                       min_child_weight=0.001, min_child_samples=10, subsample=1.0, subsample_freq=0,
#                       colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
#                       importance_type='split')

# v4
# model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=400,
#                       subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
#                       min_child_weight=0.001, min_child_samples=10, subsample=1.0, subsample_freq=0,
#                       colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
#                       importance_type='split')
# RMSE : 5.2965

# v5
# model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=475,
#                       subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
#                       min_child_weight=0.001, min_child_samples=10, subsample=1.0, subsample_freq=0,
#                       colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
#                       importance_type='split')
# RMSE :  5.2909

# v6
model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=500,
                      subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
                      min_child_weight=0.001, min_child_samples=31, subsample=1.0, subsample_freq=0,
                      colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
                      importance_type='split')
# RMSE :  5.2813

# v7
model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=475,
                      subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
                      min_child_weight=0.001, min_child_samples=31, subsample=1.0, subsample_freq=0,
                      colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
                      importance_type='split')

# 모델 학습
model.fit(train_X,train_Y)

# test 데이터 예측
pred = model.predict(test_one)

submission['INVC_CONT'] = pred

submission.to_csv('csv/submit_v7.csv', index=False)