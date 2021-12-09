import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 1000)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

x = train.iloc[:, :-1]
y = train.iloc[:, -1]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

x_train = x_train.drop(['index'], axis=1)
x_test = x_test.drop(['index'], axis=1)


# 원핫 인코딩
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# x_train = StandardScaler().fit_transform(x_train)
# x_test = StandardScaler().fit_transform(x_test)
# x_train = MinMaxScaler().fit_transform(x_train)
# x_test = MinMaxScaler().fit_transform(x_test)
# print(x_train)
# print(np.log1p(x_train))
# x_train['SEND_SPG_INNB'] = np.log1p(x_train['SEND_SPG_INNB'])
# x_test['SEND_SPG_INNB'] = np.log1p(x_test['SEND_SPG_INNB'])
# var_2 = np.expm1(var_1)


# print(x_train.info())
# print(x_train.describe())

from lightgbm import LGBMRegressor
#모델 정의
model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=400,
                      subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
                      min_child_weight=0.001, min_child_samples=10, subsample=1.0, subsample_freq=0,
                      colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
                      importance_type='split')
# 5.2965


# 모델 학습
model.fit(x_train, y_train)

# test 데이터 예측
preds = model.predict(x_test)

from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y_test, preds)**0.5

print("RMSE : ", RMSE)


# submission['INVC_CONT'] = preds
#
# submission.to_csv('baseline.csv',index = False)