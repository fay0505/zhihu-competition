#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from utils import *
from collections import Counter
import gc
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb


def trans(x):
    if x <= 0:
        return x
    if 1 <= x <= 10:
        return 1
    if 10 < x <= 100:
        return 2
    if 100 < x <= 200:
        return 3
    if 200 < x <= 300:
        return 4
    if 400 < x <= 500:
        return 5
    if 500 < x <= 600:
        return 6
    if x > 600:
        return 7


def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))


def parse_map(d):
    if d == '-1':
        return {}
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))


def most_interest_topic(d):
    if len(d) == 0:
        return -1
    return list(d.keys())[np.argmax(list(d.values()))]


def get_interest_values(d):
    if len(d) == 0:
        return [0]
    return list(d.values())


# 用户感兴趣topic和问题 topic的交集的兴趣值
def process_fun3(df):
    return df.apply(lambda row: [row['感兴趣话题'][t] for t in row['关注话题和绑定话题交集']], axis=1)


# 用户感兴趣topic和问题绑定话题的交集
def process_fun2(df):
    return df.apply(lambda row: list(set(row['感兴趣话题'].keys()) & set(row['问题绑定话题'])), axis=1)


# 用户关注topic和问题绑定话题的交集
def process_fun1(df):
    return df.apply(lambda row: list(set(row['关注话题']) & set(row['问题绑定话题'])), axis=1)


def extract_feature1(target, label_feature, ans_feature):
    # 问题特征
    t1 = label_feature.groupby('问题id')['是否回答'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['问题id', '问题_是否回答_mean', '问题_是否回答_sum', '问题_是否回答_std', '问题_是否回答_count']
    target = pd.merge(target, t1, on='问题id', how='left')


    # 用户特征
    t1 = label_feature.groupby('用户id')['是否回答'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['用户id', '用户_是否回答_mean', '用户_是否回答_sum', '用户_是否回答_std', '用户_是否回答_count']
    target = pd.merge(target, t1, on='用户id', how='left')

    # 回答部分特征
    t1 = ans_feature.groupby('问题id')['回答id'].count().reset_index()
    t1.columns = ['问题id', '问题回答总数']
    target = pd.merge(target, t1, on='问题id', how='left')

    t1 = ans_feature.groupby('用户id')['回答id'].count().reset_index()
    t1.columns = ['用户id', '用户回答总数']
    target = pd.merge(target, t1, on='用户id', how='left')

    t1 = ans_feature.groupby('用户id')['回答创建时间-day'].max().reset_index()
    t1.columns = ['用户id', '用户回答的最后时间']
    target = pd.merge(target, t1, on='用户id', how='left')



    for col in gen_feat1_answer_info_cols:
        t1 = ans_feature.groupby('用户id')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['用户id', f'用户_{col}_sum', f'用户_{col}_max', f'用户_{col}_mean']
        target = pd.merge(target, t1, on='用户id', how='left')

        t1 = ans_feature.groupby('问题id')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['问题id', f'问题_{col}_sum', f'问题_{col}_max', f'问题_{col}_mean']
        target = pd.merge(target, t1, on='问题id', how='left')
        print("extract %s", col)

    return target

# 用户知识贡献意愿的参数设置
u1 = 1.711
u2 = 1.492
p1 = 1.033
p2 = 0.075
p3 = 0.056


def fun1(x1, x2, x3):

    return np.exp(u1 - p1*x1 - p2*x2 - p3*x3) / (1 + np.exp(u1 - p1*x1 - p2*x2 - p3*x3))

def fun2(x1, x2, x3, q21):

    return np.exp(u2 - p1*x1 - p2*x2 - p3*x3) / (1 + np.exp(u2 - p1*x1 - p2*x2 - p3*x3)) - q21

def fun3(x1, x2, x3):
    return 1 -  np.exp(u2 - p1*x1 - p2*x2 - p3*x3) / (1 + np.exp(u2 - p1*x1 - p2*x2 - p3*x3))


def compute_user_willingness_to_contribute(data):

    print("计算用户的知识贡献意愿之前：", data.shape)
    data['q21'] = data.apply(lambda cols: fun1(cols['用户关注话题数'], cols['用户_回答收到的点赞数_sum'], cols['用户_回答的内容字数_sum']), axis=1)

    data['q22'] = data.apply(lambda cols: fun2(cols['用户关注话题数'], cols['用户_回答收到的点赞数_sum'], cols['用户_回答的内容字数_sum'], cols['q21']), axis=1)

    data['q23'] = data.apply(lambda cols: fun3(cols['用户关注话题数'], cols['用户_回答收到的点赞数_sum'], cols['用户_回答的内容字数_sum']), axis=1)

    print("计算用户的知识贡献意愿之后：", data.shape)

    return data

def get_feat_from_user_interest(data):
    # 用户关注和感兴趣的topic数  关注话题和感兴趣话题的统计特征，线上提升0.005
    data['关注话题'] = data['关注话题'].apply(parse_list_1)
    data['感兴趣话题'] = data['感兴趣话题'].apply(parse_map)

    data['用户关注话题数'] = data['关注话题'].apply(len)
    data['用户感兴趣话题数'] = data['感兴趣话题'].apply(len)

    # 用户最感兴趣的topic
    data['用户最感兴趣的话题'] = data['感兴趣话题'].apply(most_interest_topic)

    # 用户topic兴趣值的统计特征
    data['用户兴趣值提取'] = data['感兴趣话题'].apply(get_interest_values)
    data['用户最低兴趣值'] = data['用户兴趣值提取'].apply(np.min)
    data['用户最高兴趣值'] = data['用户兴趣值提取'].apply(np.max)
    data['用户平均兴趣值'] = data['用户兴趣值提取'].apply(np.mean)
    data['用户兴趣值方差'] = data['用户兴趣值提取'].apply(np.std)

    data.drop(['关注话题', '感兴趣话题', '用户兴趣值提取'], axis=1, inplace=True)

    return data


def lgb_predict(X_train, y_train, X_test, kFolds=5, use_cv=True, get_importance=False):
    model_lgb = LGBMClassifier(n_estimators=2000,
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

        model_lgb.fit(X_train.values, y_train.values,
                      eval_names=['train'],
                      eval_metric=['logloss', 'auc'],
                      eval_set=[(X_train.values, y_train.values)],
                      early_stopping_rounds=50)
        y_pred = model_lgb.predict_proba(X_test.values)[:, 1]

        return y_pred


def xgb_predict(X_train, y_train, X_test, kFolds=5, use_cv=True, eval_set=None):

    params = {

        'objective': 'binary:logistic',

        'seed': 1000,
        'n_estimators': 3000,
        'scale_pos_weight':5,
        'tree_method': 'gpu_hist',
    }
    xgb_model = xgb.XGBClassifier(**params)
    if use_cv:
        print("use cv..")
        kfold = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=2019)
        kf = kfold.split(X_train, y_train)
        cv_y_pred = np.zeros(X_test.shape[0])

        for i, (train_batch, val_batch) in enumerate(kf):
            x_train_batch, x_val_batch, y_train_batch, y_val_batch = \
                X_train.iloc[train_batch, :], X_train.iloc[val_batch, :], \
                y_train.iloc[train_batch], y_train.iloc[val_batch]



            xgb_model.fit(x_train_batch, y_train_batch,
                          eval_set=[(x_val_batch, y_val_batch)], eval_metric=['logloss', 'auc'],
                          early_stopping_rounds=200)


            prob = xgb_model.predict_proba(X_test.values)[:,1]
            print(prob)
            tmp = prob[:,1]
            cv_y_pred += tmp

        cv_y_pred /= kFolds
        return cv_y_pred
    else:
        if eval_set is not None:
            x_val, y_val = eval_set
            xgb_model.fit(X_train.values, y_train.values,
                          eval_set=[(x_val.values, y_val.values)], eval_metric=['logloss', 'auc'],
                          early_stopping_rounds=200)
            val_pred = xgb_model.predict_proba(x_val.values)
            np.save('xgb_val_pred2d.npy', val_pred)
            test_pred = xgb_model.predict_proba(X_test.values)
            np.save('xgb_test_pred2d.npy', test_pred)

            return test_pred[:,1]

# 导入数据
user_info = pd.read_csv('../../data_set/member_info_0926.txt', header=None, sep='\t')
user_info.columns = ['用户id', '性别', '创作关键词', '创作数量等级', '创作热度等级', '注册类型', '注册平台', '访问频率', '用户二分类特征a', '用户二分类特征b',
                     '用户二分类特征c', '用户二分类特征d', '用户二分类特征e', '用户多分类特征a', '用户多分类特征b', '用户多分类特征c', '用户多分类特征d', '用户多分类特征e',
                     '盐值', '关注话题', '感兴趣话题']
user_info.drop(['创作关键词', '创作数量等级', '创作热度等级', '注册类型', '注册平台'], axis=1, inplace=True)
user_info = get_feat_from_user_interest(user_info)

question_info = pd.read_csv('../../data_set/question_info_0926.txt', header=None, sep='\t')
question_info.columns = ['问题id', '问题创建时间', '问题标题单字编码', '问题标题切词编码', '问题描述单字编码', '问题描述切词编码', '问题绑定话题']
question_info.drop(['问题标题单字编码', '问题标题切词编码', '问题描述单字编码', '问题描述切词编码'], axis=1, inplace=True)

# ['问题id', '用户id', '邀请创建时间', '是否回答', 'attend_topic_question_cos',
#        'interest_topic_question_cos', 'most_interest_topic_question_cos']
train = pd.read_csv("../feat/train_cos_feat.csv")
test = pd.read_csv("../feat/test1_cos_feat.csv")
print("train:{0}, test:{1}, \n columns:{2}".format(train.shape, test.shape, train.columns))

sub = test[['问题id', '用户id', '邀请创建时间']].copy()
sub_size = len(sub)

train['邀请创建时间-day'] = train['邀请创建时间'].apply(lambda x: x.split('-')[0].split('D')[1]).astype(int)
train['邀请创建时间-hour'] = train['邀请创建时间'].apply(lambda x: x.split('-')[1].split('H')[1]).astype(int)

test['邀请创建时间-day'] = test['邀请创建时间'].apply(lambda x: x.split('-')[0].split('D')[1]).astype(int)
test['邀请创建时间-hour'] = test['邀请创建时间'].apply(lambda x: x.split('-')[1].split('H')[1]).astype(int)

question_info['问题创建时间-day'] = question_info['问题创建时间'].apply(lambda x: x.split('-')[0].split('D')[1]).astype(int)
question_info['问题创建时间-hour'] = question_info['问题创建时间'].apply(lambda x: x.split('-')[1].split('H')[1]).astype(int)

del train['邀请创建时间'], test['邀请创建时间'], question_info['问题创建时间']
gc.collect()

answer_info = pd.read_csv('../../data_set/answer_info_0926.txt', header=None, sep='\t')
answer_info.columns = answer_info_columns
answer_info.drop(['回答内容单字编码', '回答内容切词编码', '回答是否被收入圆桌'], axis=1, inplace=True)
answer_info['回答创建时间-day'] = answer_info['回答创建时间'].apply(lambda x: x.split('-')[0].split('D')[1]).astype(int)
answer_info['回答创建时间-hour'] = answer_info['回答创建时间'].apply(lambda x: x.split('-')[1].split('H')[1]).astype(int)

answer_info = pd.merge(answer_info, question_info, on='问题id')

del question_info
gc.collect()

# 时间窗口划分
# train
# val
train_start = 3838
train_end = 3867

val_start = 3868
val_end = 3874

label_end = 3867
label_start = label_end - 6

train_label_feature_end = label_end - 7
train_label_feature_start = train_label_feature_end - 22

train_ans_feature_end = label_end - 7
train_ans_feature_start = train_ans_feature_end - 50

val_label_feature_end = val_start - 1
val_label_feature_start = val_label_feature_end - 22

val_ans_feature_end = val_start - 1
val_ans_feature_start = val_ans_feature_end - 50

train_label_feature = train[
    (train['邀请创建时间-day'] >= train_label_feature_start) & (train['邀请创建时间-day'] <= train_label_feature_end)]
print("train_label_feature %s", train_label_feature.shape)

val_label_feature = train[
    (train['邀请创建时间-day'] >= val_label_feature_start) & (train['邀请创建时间-day'] <= val_label_feature_end)]
print("val_label_feature %s", val_label_feature.shape)

train_label = train[(train['邀请创建时间-day'] > train_label_feature_end)]

print("train feature start {0} end {1}, label start {2} end {3}".format( train_label_feature['邀请创建时间-day'].min(),
      train_label_feature['邀请创建时间-day'].max(), train_label['邀请创建时间-day'].min(), train_label['邀请创建时间-day'].max()))

print("test feature start {0} end {1}, label start {2} end {3}".format(val_label_feature['邀请创建时间-day'].min(),
      val_label_feature['邀请创建时间-day'].max(), test['邀请创建时间-day'].min(), test['邀请创建时间-day'].max()))

# 确定ans的时间范围
# 3807~3874
train_ans_feature = answer_info[
    (answer_info['回答创建时间-day'] >= train_ans_feature_start) & (answer_info['回答创建时间-day'] <= train_ans_feature_end)]

val_ans_feature = answer_info[
    (answer_info['回答创建时间-day'] >= val_ans_feature_start) & (answer_info['回答创建时间-day'] <= val_ans_feature_end)]

print("train ans feature %s, start %s end %s", train_ans_feature.shape, train_ans_feature['回答创建时间-day'].min(),
      train_ans_feature['回答创建时间-day'].max())

print("val ans feature %s, start %s end %s", val_ans_feature.shape, val_ans_feature['回答创建时间-day'].min(),
      val_ans_feature['回答创建时间-day'].max())

train_label = extract_feature1(train_label, train_label_feature, train_ans_feature)
test = extract_feature1(test, val_label_feature, val_ans_feature)

print("train shape %s, test shape %s", train_label.shape, test.shape)
assert len(test) == sub_size

train_label = pd.merge(train_label, user_info, how='left', on='用户id')
test = pd.merge(test, user_info, how='left', on='用户id')

print("合并user info之后：", train_label.shape, test.shape)

# 数据合并
data = pd.concat([train_label, test], axis=0, sort=True)
print("112 数据合并之后特征：", data.columns)

#data = compute_user_willingness_to_contribute(data)

train_samples = train_label.shape[0]
del train_label, test
gc.collect()

# 读入一些别的做好的特征
user_category = pd.read_csv('../feat/user_category.csv')
data = pd.merge(data, user_category, on='用户id', how='left')
del user_category
gc.collect()

question_feats = ['问题id'] + ["topic_vec{0}".format(i) for i in range(64)] + ["t_w_vec{0}".format(i) for i in range(64)]
question_embed_topic = pd.read_csv('../feat/question_all_avg_embed.csv')[question_feats]
print(" 合并question_embed_topic 之前：", data.shape)
data = pd.merge(data, question_embed_topic, on='问题id', how='left')
print("合并question_embed_topic 之后：", data.shape)

del question_embed_topic
gc.collect()

question_category_feat = pd.read_csv("../feat/question_feat.csv")
print("合并question category feat之前:", data.shape, question_category_feat.shape)
data = pd.merge(data, question_category_feat, on='问题id', how='left')
print("合并question category feat之后:", data.shape, question_category_feat.shape)



del question_category_feat
gc.collect()
print("300 特征合并之后特征：", data.columns)
user_feat = pd.read_csv('../feat/part_user_receive_likes_question_topicVec.csv')

data = pd.merge(data, user_feat, on='用户id', how='left')
data.fillna(0, inplace=True)

print("合并擅长领域之后：", data.shape)
del user_feat


print("saving data...")
data[:train_samples].to_csv('train.csv', index=False)
data[train_samples:].to_csv('test.csv', index=False)

drop_feat = [

    '用户二分类特征e', '用户二分类特征c',
    '用户二分类特征b', '用户二分类特征a', '用户二分类特征d',
    '用户多分类特征e', ]
data.drop(drop_feat, axis=1, inplace=True)

print("300 特征合并之后特征：", data.columns)
# **编码：**将离散型的特征通过LabelEncoder进行数字编码。
class_feat = ['用户id', '问题id', '性别', '访问频率', '用户多分类特征a', '用户多分类特征b', '用户多分类特征c', '用户多分类特征d',  '用户最感兴趣的话题']
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

# 压缩数据
t = data.dtypes
for x in t[t == 'int64'].index:
    data[x] = data[x].astype('int32')

for x in t[t == 'float64'].index:
    data[x] = data[x].astype('float32')


data.fillna(0, inplace=True)
# 划分训练集和测试集
y_train = data[:train_samples]['是否回答']
X_train = data[:train_samples]
X_train['weight'] = X_train['是否回答'].apply(lambda x : 1 if x == 0 else 2 )
X_train.drop(['是否回答'], axis=1, inplace=True)

X_test = data[train_samples:].drop(['是否回答'], axis=1)
# 用于保存提交结果

print("use these cols to train...\n ", data.columns)





y_pred = lgb_predict(X_train, y_train, X_test, use_cv=True)
sub['是否回答'] = y_pred
sub.to_csv('result.txt', index=False, header=False, sep='\t')
