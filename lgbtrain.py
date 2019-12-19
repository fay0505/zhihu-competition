import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from utils import *
import lightgbm as lgb



def lgb_predict(X_train, y_train, X_test, kFolds=5, use_cv=True, eval_set=None):
    model_lgb = LGBMClassifier(n_estimators=2000, scale_pos_weight=5,
                               objective='binary', seed=1000, n_jobs=-1, silent=True)

    params = {

        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'seed': 1000,
    }
    if use_cv:
        kfold = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=2019)
        kf = kfold.split(X_train, y_train)
        cv_y_pred = np.zeros(X_test.shape[0])

        for i, (train_batch, val_batch) in enumerate(kf):
            x_train_batch, x_val_batch, y_train_batch, y_val_batch = \
                X_train.iloc[train_batch, :], X_train.iloc[val_batch, :], \
                y_train.iloc[train_batch], y_train.iloc[val_batch]

            tr_weight = x_train_batch['weight']
            te_weight = x_val_batch['weight']
            del x_train_batch['weight'], x_val_batch['weight']

            lgb_train = lgb.Dataset(x_train_batch.values, y_train_batch.values, weight=tr_weight)
            lgb_val = lgb.Dataset(x_val_batch.values, y_val_batch.values, weight=te_weight)

            gbm = lgb.train(params, lgb_train, valid_sets=lgb_val, early_stopping_rounds=200, num_boost_round=2000)
            tmp=  gbm.predict(X_test.values, num_iteration=gbm.best_iteration)

            cv_y_pred += tmp

        cv_y_pred /= kFolds

        return cv_y_pred

    else:
        if eval_set is not None:

            model_lgb.fit(X_train.values, y_train.values, eval_set=[(x_val.values, y_val.values)],
                          eval_metric=['logloss', 'auc'])
            # tr_weight = X_train['weight']
            # te_weight = x_val['weight']
            # del X_train['weight'], x_val['weight']
            #
            # lgb_train = lgb.Dataset(X_train.values, y_train.values, weight=tr_weight)
            # lgb_val = lgb.Dataset(x_val.values, y_val.values, weight=te_weight)
            #
            # gbm = lgb.train(params, lgb_train, valid_sets=lgb_val, early_stopping_rounds=200, num_boost_round=2000)
            test_pre = model_lgb.predict_proba(X_test.values, num_iteration=model_lgb.best_iteration_)
            np.save('lgb_test_pre2d.npy', test_pre)
            val_pre = model_lgb.predict_proba(x_val.values, num_iteration=model_lgb.best_iteration_)
            np.save('lgb_val_pre2d.npy', val_pre)

            return test_pre[:,1]

if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')


    ans = pd.read_csv('result_lgb.txt', header=None, sep='\t')
    ans.columns = ['问题id', '用户id', '邀请创建时间', "是否回答"]


    print(train_data.shape, test_data.shape)
    data = pd.concat([train_data, test_data], axis=0, sort=True)



    class_feat = ['用户id', '问题id', '性别', '访问频率', '用户多分类特征a', '用户多分类特征b', '用户多分类特征c', '用户多分类特征d','用户最感兴趣的话题']
    encoder = LabelEncoder()
    for feat in class_feat:
        encoder.fit(data[feat])
        data[feat] = encoder.transform(data[feat])

    # **构造计数特征：**对具有很好区分度的特征进行单特征计数(有明显提升)。

    for feat in ['用户id', '问题id',  '访问频率', '用户多分类特征a', '用户多分类特征b', '用户多分类特征c', '用户多分类特征d']:
        col_name = '{}_count'.format(feat)
        data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
        data.loc[data[col_name] < 2, feat] = -1
        data[feat] += 1
        data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
        data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())



    data.drop(drop_feat, axis=1, inplace=True)

    result_append = ans[['问题id', '用户id', '邀请创建时间']]

    use_cv = False
    if use_cv:
        y_train = data[:train_data.shape[0]]['是否回答']
        X_train = data[:train_data.shape[0]]

        X_test = data[train_data.shape[0]:].drop(['是否回答'], axis=1)
        X_train.drop(['是否回答'], axis=1, inplace=True)
        print("use these cols to train...\n ", data.columns)
        y_pred = lgb_predict(X_train, y_train, X_test, use_cv=True)
    else:
        train = data[:train_data.shape[0]]
        # train['weight'] = train['是否回答'].apply(lambda x: 1 if x == 0 else 5)
        x_train = train[(train['邀请创建时间-day'] >= 3861) & (train['邀请创建时间-day'] <= 3866)]
        x_val = train[(train['邀请创建时间-day'] == 3867)]

        del train
        y_train = x_train['是否回答']
        y_val = x_val['是否回答']
        x_train.drop(['是否回答'], axis=1, inplace=True)
        x_val.drop(['是否回答'], axis=1, inplace=True)

        X_test = data[train_data.shape[0]:].drop(['是否回答'], axis=1)

        print("use these cols to train...\n ", data.columns)
        y_pred = lgb_predict(x_train, y_train, X_test, use_cv=False, eval_set=(x_val, y_val))

    result_append['是否回答'] = y_pred
    result_append.to_csv('result_lgb1.txt', index=False, header=False, sep='\t')