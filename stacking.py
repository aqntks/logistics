import numpy as np

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


# 개별 기반 모델에서 최종 메타 모델이 사용할 학습 및 테스트용 데이터를 생성하기 위한 함수.
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    # 지정된 n_folds값으로 KFold 생성.
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    # 추후에 메타 모델이 사용할 학습 데이터 반환을 위한 넘파이 배열 초기화
    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    print(model.__class__.__name__, ' model 시작 ')

    for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n)):
        # 입력된 학습 데이터에서 기반 모델이 학습/예측할 폴드 데이터 셋 추출
        print('\t 폴드 세트: ', folder_counter, ' 시작 ')
        X_tr = X_train_n[train_index]
        y_tr = y_train_n[train_index]
        X_te = X_train_n[valid_index]

        # 폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행.
        model.fit(X_tr, y_tr)
        # 폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장.
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1, 1)
        # 입력된 원본 테스트 데이터를 폴드 세트내 학습된 기반 모델에서 예측 후 데이터 저장.
        test_pred[:, folder_counter] = model.predict(X_test_n)

    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)

    # train_fold_pred는 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터
    return train_fold_pred, test_pred_mean


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


X_train_n = train_X.values
X_test_n = test_one.values
y_train_n = train_Y.values

model1 = RandomForestRegressor()

model2 = GradientBoostingRegressor()

model3 = XGBRegressor()

model4 = LGBMRegressor()

model5 = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=400,
                      subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
                      min_child_weight=0.001, min_child_samples=10, subsample=1.0, subsample_freq=0,
                      colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
                      importance_type='split')

train1, test1 = get_stacking_base_datasets(model1, X_train_n, y_train_n, X_test_n, 5)
train2, test2 = get_stacking_base_datasets(model2, X_train_n, y_train_n, X_test_n, 5)
train3, test3 = get_stacking_base_datasets(model3, X_train_n, y_train_n, X_test_n, 5)
train4, test4 = get_stacking_base_datasets(model4, X_train_n, y_train_n, X_test_n, 5)
train5, test5 = get_stacking_base_datasets(model5, X_train_n, y_train_n, X_test_n, 5)

# 개별 모델이 반환한 학습 및 테스트용 데이터 세트를 Stacking 형태로 결합.
Stack_final_X_train = np.concatenate((train1, train2,
                                      train3, train4, train5), axis=1)
Stack_final_X_test = np.concatenate((test1, test2,
                                     test3, test4, test5), axis=1)

# 최종 메타 모델은 라쏘 모델을 적용.
meta_model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=400,
                      subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.2,
                      min_child_weight=0.001, min_child_samples=10, subsample=1.0, subsample_freq=0,
                      colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
                      importance_type='split')

#기반 모델의 예측값을 기반으로 새롭게 만들어진 학습 및 테스트용 데이터로 예측하고 RMSE 측정.
meta_model.fit(Stack_final_X_train, train_Y)
pred = meta_model.predict(Stack_final_X_test)


submission['INVC_CONT'] = pred

submission.to_csv('csv/stacking_v2.csv',index = False)