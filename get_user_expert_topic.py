import pandas as pd
import  numpy as np
from utils import *
import gc


##### 统计用户的擅长领域，计算用户收到点赞数、收藏数、感谢数最多的问题的话题绑定向量
# 在实际使用中，只使用了用户收到点赞数最多的问题的话题绑定向量
# 另外再计算过用户擅长领域 和 题目的切词、绑定话题 向量cos距离，没有实际提升，故最后未使用

def get_feat(answer_info, name='train'):
    # 寻找用户擅长领域
    # 求出用户收到的最大点赞数、感谢数、回答收藏数对应的问题id
    max_index = answer_info['回答收到的点赞数'].groupby(answer_info['用户id']).idxmax()
    print("点赞数最多 index:", max_index)
    feat1 = answer_info[['用户id','问题id']].loc[max_index.values]
    feat1.columns = ['用户id','点赞数最多的问题id' ]

    max_index = answer_info['回答收到的感谢数'].groupby(answer_info['用户id']).idxmax()
    print("感谢数最多 index:", max_index)
    feat2 =  answer_info[['用户id','问题id']].loc[max_index.values]
    feat2.columns = ['用户id', '感谢数最多的问题id']

    max_index = answer_info['回答收藏数'].groupby(answer_info['用户id']).idxmax()
    print("收藏数最多 index:", max_index)
    feat3 = answer_info[['用户id','问题id']].loc[max_index.values]
    feat3.columns = ['用户id', '收藏数最多的问题id']

    feat = pd.merge(feat1, feat2, on='用户id', how='left')
    feat = pd.merge(feat, feat3, on='用户id', how='left')

    del feat1, feat2, feat3
    gc.collect()


    print("用户收到的最大点赞数、感谢数、回答收藏数对应的问题id 已计算完毕！\n feat shape:", feat.shape)
    print(feat.head(5))


    feat_cols = ["topic_vec{0}".format(i) for i in range(64)]
    topic_vectors = pd.read_csv('../feat/question_all_avg_embed.csv')
    topic_ids = topic_vectors['问题id'].tolist()
    topic_vectors = topic_vectors[feat_cols]
    print("topic vectors shape:", topic_vectors.shape)

    topic_vectors = topic_vectors.T
    topic_vectors.columns = topic_ids
    print("topic vectors shape:", topic_vectors.shape)

    feat1_vec = []
    feat2_vec = []
    feat3_vec = []

    num_samples = feat.shape[0]
    zero_vec = [0.0] * 64
    for index, row in feat.iterrows():
        feat1_question_id = row['点赞数最多的问题id']
        feat2_question_id = row['感谢数最多的问题id']
        feat3_question_id = row['收藏数最多的问题id']
        print("index:{0}, finish:{1}, 点赞数最多的问题:{2}, 感谢数最多的问题:{3},收藏数最多的问题:{4}".
              format(index, (index+1)/num_samples, feat1_question_id, feat2_question_id, feat3_question_id))

        if feat1_question_id == np.nan:
            feat1_vec.append(zero_vec)
        else:
            feat1_vec.append(topic_vectors[feat1_question_id].tolist())

        if feat2_question_id == np.nan:
            feat2_vec.append(zero_vec)
        else:
            feat2_vec.append(topic_vectors[feat2_question_id].tolist())

        if feat3_question_id == np.nan:
            feat3_vec.append(zero_vec)
        else:
            feat3_vec.append(topic_vectors[feat3_question_id].tolist())

    feat1_df = pd.DataFrame(feat1_vec, columns=["点赞数最多的问题的topicVec_{0}".format(i) for i in range(64)])
    feat2_df = pd.DataFrame(feat2_vec, columns=["感谢数最多的问题的topicVec_{0}".format(i) for i in range(64)])
    feat3_df = pd.DataFrame(feat3_vec, columns=["收藏数最多的问题的topicVec_{0}".format(i) for i in range(64)])

    print(feat1_df.shape, feat2_df.shape, feat3_df.shape)
    feat1_df['用户id'] = feat['用户id']
    feat1_df.to_csv('../feat/{0}_user_receive_likes_question_topicVec.csv'.format(name), index=False)

    feat2_df['用户id'] = feat['用户id']
    feat2_df.to_csv('../feat/{0}_user_receive_thanks_question_topicVec.csv'.format(name), index=False)

    feat3_df['用户id'] = feat['用户id']
    feat3_df.to_csv('../feat/{0}_user_receive_collects_question_topicVec.csv'.format(name), index=False)

    print(feat1_df.head(5))

answer_info = pd.read_csv('../../data_set/answer_info_0926.txt', header=None, sep='\t')
answer_info.columns = answer_info_columns
answer_info = answer_info[['问题id', '用户id','回答创建时间', '回答收到的点赞数', '回答收到的感谢数', '回答收藏数']]
answer_info['回答创建时间-day'] = answer_info['回答创建时间'].apply(lambda x: x.split('-')[0].split('D')[1]).astype(int)
answer_info = answer_info[(answer_info['回答创建时间-day'] >= 3817) & (answer_info['回答创建时间-day'] <= 3867)]
del answer_info['回答创建时间'], answer_info['回答创建时间-day']
get_feat(answer_info, 'test')
exit()

answer_info = answer_info[(answer_info['回答创建时间-day'] >= 3810) & (answer_info['回答创建时间-day'] <= 3860)]
del answer_info['回答创建时间'], answer_info['回答创建时间-day']

print("answer info shape:", answer_info.shape)
print(answer_info.head(5))
unique_user = set(list(np.unique(answer_info['用户id'])))



train =  pd.read_csv('../../data_set/invite_info_0926.txt', header=None, sep='\t')
train.columns = ['问题id', '用户id', '邀请创建时间','是否回答']
print("train:", train.shape)
train = train[train['是否回答'] == 1]
print("train 正样本数：", train.shape)
train['邀请创建时间-day'] = train['邀请创建时间'].apply(lambda x: x.split('-')[0].split('D')[1]).astype(int)
train = train[(train['邀请创建时间-day'] >= 3838) & (train['邀请创建时间-day'] <= 3860)]
del train['邀请创建时间-day'], train['是否回答'], train['邀请创建时间']
print("train 筛选日期之后: ",train.shape)


train = pd.merge(train, answer_info, how='left', on=['用户id', '问题id'])
print("train 和 answer 合并之后:", train.shape)
print(train.head(5))
for col in train.columns:

    print("合并之后{0}是否存在nan:{1}".format(col, train.isnull().any().any()))

print("删除存在nan得行")
train.dropna(axis=0, how='any',  inplace=True)
train = train.reset_index(drop=True)
print("缺失值处理之后", train.shape)
for col in train.columns:

    print("删除缺失值之后{0}是否存在nan:{1}".format(col, train.isnull().any().any()))
del answer_info
get_feat(train, 'part')


