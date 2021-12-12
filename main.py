import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 1000)

mode = 1

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

print(train)
print(train.info())
print(train.describe())
print(train['DL_GD_LCLS_NM'].value_counts())
print(train['DL_GD_MCLS_NM'].value_counts())

x = train.iloc[:, :-1]
y = train.iloc[:, -1]

# 정규분포
# y = np.log1p(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

x_train = x_train.drop(['index'], axis=1)
x_test = x_test.drop(['index'], axis=1)


# 원핫 인코딩
columns = ['SEND_SPG_INNB', 'REC_SPG_INNB', 'DL_GD_LCLS_NM', 'DL_GD_MCLS_NM']
# x_train = pd.get_dummies(x_train, columns=columns)
# x_test = pd.get_dummies(x_test, columns=columns)
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# print(x_train)
#
# for col in x_train.columns:
#     if col in x_test.columns:
#         pass
#     else:
#         x_test[col] = 0
#
# for col in x_test.columns:
#     if col in x_train.columns:
#         pass
#     else:
#         x_train[col] = 0


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

if mode:

    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn import svm
    #모델 정의

    # model = svm.SVR()
    # RMSE: 6.427058263306755

    # model = LinearRegression(normalize=True)
    # RMSE : 6.241782013683741
    # RMSE : normalize=True 6.122783806462542

    # model = LogisticRegression()
    # RMSE :  6.496585641704417

    # model = RandomForestRegressor()
    # RMSE :  5.5472239703993536

    # model = DecisionTreeRegressor()
    # RMSE :  5.812296994047502

    # model = GradientBoostingRegressor()
    # RMSE :  5.59496674573263

    # model = Ridge()
    # RMSE :  6.123063773037983

    # model = Lasso()
    # RMSE :  6.217901010430412

    # model = XGBRegressor(n_estimators=1000, min_split_gain=0.2, colsample_bytree=0.9)
    # RMSE :  5.494675212439629

    # model = LGBMRegressor()
    # RMSE :  5.3499672075937115


    # boosting_type = 'gbdt', num_leaves = 31, max_depth = - 1, learning_rate = 0.1, n_estimators = 100,
    # subsample_for_bin = 200000, objective = None, class_weight = None, min_split_gain = 0.0,
    # min_child_weight = 0.001, min_child_samples = 20, subsample = 1.0, subsample_freq = 0,
    # colsample_bytree = 1.0, reg_alpha = 0.0, reg_lambda = 0.0, random_state = None,
    # n_jobs = - 1, importance_type = 'split

    model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=1000,
                          subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
                          min_child_weight=0.001, min_child_samples=11, subsample=1.0, subsample_freq=0,
                          colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
                          importance_type='split')
    # RMSE : 5.2965
    # 5.2909

    # 모델 학습
    evals = [(x_test, y_test)]
    model.fit(x_train, y_train, eval_metric="rmse", eval_set=evals, verbose=True)
    # model.fit(x_train, y_train)
    # test 데이터 예측
    preds = model.predict(x_test)

    # 정규분포 풀기
    # y_test = np.expm1(y_test)
    # preds = np.expm1(preds)
    from sklearn.metrics import mean_squared_error

    RMSE = mean_squared_error(y_test, preds) ** 0.5

    print("RMSE : ", RMSE)

    # bestP3 = 1000
    # bestRMSE = 100
    # for p3 in range(0, 100):
    #     model = LGBMRegressor(n_estimators=500, min_split_gain=0.2, colsample_bytree=0.9, min_child_samples=p3)
    #
    #     # 모델 학습
    #     evals = [(x_test, y_test)]
    #     model.fit(x_train, y_train, eval_metric="rmse", eval_set=evals, verbose=True)
    #
    #     # test 데이터 예측
    #     preds = model.predict(x_test)
    #
    #     # 정규분포 풀기
    #     # y_test = np.expm1(y_test)
    #     # preds = np.expm1(preds)
    #     from sklearn.metrics import mean_squared_error
    #     RMSE = mean_squared_error(y_test, preds)**0.5
    #
    #     print("RMSE : ", RMSE)
    #     if RMSE < bestRMSE:
    #         bestRMSE = RMSE
    #         bestP3 = p3
    # print(f'bestRMSE - {bestRMSE}, p3 - {bestP3}')


    # bestRMSE = 100
    # bestP1 = 0.0
    # bestP2 = 0.0
    # bestP3 = 3
    # for p1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     for p2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #         for p3 in [3, 5, 7, 9, 10, 12, 14, 16, 18, 20]:
    #             model = LGBMRegressor(n_estimators=500, min_split_gain=p1, colsample_bytree=p2, min_child_samples=p3)
    #
    #             # 모델 학습
    #             evals = [(x_test, y_test)]
    #             model.fit(x_train, y_train, eval_metric="rmse", eval_set=evals, verbose=True)
    #
    #             # test 데이터 예측
    #             preds = model.predict(x_test)
    #
    #             # 정규분포 풀기
    #             # y_test = np.expm1(y_test)
    #             # preds = np.expm1(preds)
    #             from sklearn.metrics import mean_squared_error
    #             RMSE = mean_squared_error(y_test, preds)**0.5
    #
    #             print("RMSE : ", RMSE)
    #             if RMSE < bestRMSE:
    #                 bestRMSE = RMSE
    #                 bestP1 = p1
    #                 bestP2 = p2
    #                 bestP3 = p3
    # print(f'bestRMSE - {bestRMSE}, p1 - {bestP1}, p2 - {bestP2}, p3 - {bestP3}')

    # GridSearchCV
    # from sklearn.model_selection import GridSearchCV
    #
    # model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=500,
    #                       subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0,
    #                       min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0,
    #                       colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
    #                       importance_type='split')
    #
    # # params = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    # #           'num_leaves': [11, 15, 21, 25, 31, 35, 41, 45],
    # #           'max_depth': [-1, 128, 160],
    # #           'learning_rate': [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.11, 0.12, 0.13, 0.14, 0.15],
    # #           'subsample_for_bin': [150000, 200000, 160000, 180000, 210000, 220000],
    # #           'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # #           'min_child_samples': [3, 5, 7, 9, 10, 12, 14, 16, 18, 20],
    # #           'min_child_weight': [0.001, 0.0025, 0.00025],
    # #           'subsample': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # #           'subsample_freq': [0],
    # #           'colsample_bytree': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # #           'reg_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # #           'reg_lambda': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # #           }
    # params = {
    #           'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #           'min_child_samples': [3, 5, 7, 9, 10, 12, 14, 16, 18, 20],
    #           'colsample_bytree': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #           }
    #
    # gridcv = GridSearchCV(model, param_grid=params, cv=5, scoring='neg_mean_squared_error')
    # gridcv.fit(x_train, y_train)
    # rmse = np.sqrt(-1 * gridcv.best_score_)
    #
    # print(f'rmse -> {rmse} best -> ', gridcv.best_params_)



    # submission['INVC_CONT'] = preds
    #
    # submission.to_csv('baseline.csv',index = False)