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

# v3
model1 = RandomForestRegressor()
# RMSE :  5.5472239703993536
model1.fit(train_X,train_Y)
pred1 = model1.predict(test_one)
w1 = 10 - 5.5472

model2 = GradientBoostingRegressor()
# RMSE :  5.59496674573263
model2.fit(train_X,train_Y)
pred2 = model2.predict(test_one)
w2 = 10 - 5.5949

model3 = XGBRegressor()
# RMSE :  5.494675212439629
model3.fit(train_X,train_Y)
pred3 = model3.predict(test_one)
w3 = 10 - 5.4946

model4 = LGBMRegressor()
# RMSE :  5.3499672075937115
model4.fit(train_X,train_Y)
pred4 = model4.predict(test_one)
w4 = 10 - 5.3499

model5 = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=400,
                      subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
                      min_child_weight=0.001, min_child_samples=10, subsample=1.0, subsample_freq=0,
                      colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
                      importance_type='split')
# RMSE : 5.2965
model5.fit(train_X,train_Y)
pred5 = model5.predict(test_one)
w5 = 10 - 5.2965

all_preds = (pred3 * w3) + (pred4 * w4) + (pred5 * w5)
results = all_preds / (w3 + w4 + w5)
pred = results

# 모델 학습
# model.fit(train_X,train_Y)
#
# # test 데이터 예측
# pred = model.predict(test_one)

submission['INVC_CONT'] = pred

submission.to_csv('csv/ensemble_v3.csv',index = False)