import pandas as pd
from utils import *
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import gc


def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: str(x), str(d).split(',')))


def parse_map(d):
    if d == '-1':
        return {}
    return dict([z.split(':')[0], z.split(':')[1]] for z in d.split(','))


def most_interest_topic(d):
    if len(d) == 0:
        return [0]
    return [list(d.keys())[np.argmax(list(d.values()))]]


def get_interest_topics(d):
    if len(d) == 0:
        return [0]
    return list(d.keys())


def parse_str(d):
    return np.array(list(map(float, d.split())))


def cal_avg_embedding(data, ids):
    # topic 缺少，范围64d 0向量
    if len(ids) == 1 and ids[0] == 0:
        return [0] * 64

    ans = data[ids].values.mean(axis=1)

    ans = ans[0].tolist()
    assert len(ans) == 64
    return  ans

### 计算question中问题标题和描述的单字编码、切词编码的平均embed



question_info = pd.read_csv("../../data_set/question_info_0926.txt", header=None, sep='\t')
question_info.columns = ['问题id','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']
question_info.drop(['问题绑定话题'], axis=1, inplace=True)
print("question info shape:", question_info.shape)
print("解析之前:\n", question_info.head(5))
for col in question_info.columns:
    if col not in ['问题id','问题创建时间']:
        print("解析{0}...".format(col))
        question_info[col] = question_info[col].apply(parse_list_1)
print("解析之后:\n", question_info.head(5))

single_word_vec = pd.read_csv('../../data_set/single_word_vectors_64d.txt', sep='\t', header=None, names=['单字编码id', 'embed'])
single_word_vec['embed'] = single_word_vec['embed'].apply(parse_str)
swids = single_word_vec['单字编码id'].tolist()
single_word_vec.drop(['单字编码id'], axis=1, inplace=True)
single_word_vec = single_word_vec.T
single_word_vec.columns = swids
single_word_vec.to_csv("../feat/dealed_single_word.csv", index=False)

word_vec = pd.read_csv('../../data_set/word_vectors_64d.txt', sep='\t', header=None, names=["单词编码id", 'embed'])
word_vec['embed'] = word_vec['embed'].apply(parse_str)
word_ids = word_vec['单词编码id'].tolist()
word_vec.drop(['单词编码id'], axis=1, inplace=True)
word_vec = word_vec.T
word_vec.columns = word_ids
word_vec.to_csv("../feat/dealed_word_vec.csv", index=False)
print("finish preprocessing the single word vectors and word vectors")

question_info = pd.read_csv("../../data_set/question_info_0926.txt", header=None, sep='\t')
question_info.columns = ['问题id','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']
question_info.drop(['问题创建时间', '问题绑定话题'], axis=1, inplace=True)
print("question info shape:", question_info.shape)
print("解析编码之前:\n", question_info.head(5))

for col in question_info.columns:
    if col != '问题id':
        question_info[col] = question_info[col].apply(parse_list_1)

print("解析绑定话题之后:\n", question_info.head(5))
# 计算问题的单字、单词编码的平均值
title_sw_avg_embedding = []
title_w_avg_embedding = []
describle_sw_avg_embedding = []
describle_w_avg_embedding = []
for index, row in question_info.iterrows():
   print(index)
   title_sw_avg_embedding.append(cal_avg_embedding(single_word_vec, row['问题标题单字编码']))
   title_w_avg_embedding.append(cal_avg_embedding(word_vec, row['问题标题切词编码']))

   describle_sw_avg_embedding.append(cal_avg_embedding(single_word_vec, row['问题描述单字编码']))
   describle_w_avg_embedding.append(cal_avg_embedding(word_vec, row['问题描述切词编码']))

title_sw_avg_embed_df = pd.DataFrame(title_sw_avg_embedding, columns=["t_sw_vec{0}".format(i) for i in range(64)])
title_w_avg_embed_df = pd.DataFrame(title_w_avg_embedding, columns=["t_w_vec{0}".format(i) for i in range(64)])
describle_sw_avg_embed_df = pd.DataFrame(describle_sw_avg_embedding, columns=["d_sw_vec{0}".format(i) for i in range(64)])
describle_w_avg_embed_df = pd.DataFrame(describle_w_avg_embedding, columns=["d_w_vec{0}".format(i) for i in range(64)])

del title_sw_avg_embedding, title_w_avg_embedding, describle_sw_avg_embedding, describle_w_avg_embedding
gc.collect()

question_info = pd.concat([question_info['问题id'], title_sw_avg_embed_df, title_w_avg_embed_df, describle_sw_avg_embed_df, describle_w_avg_embed_df], axis=1)

print("计算完毕每个问题的embedding, question info shape:", question_info.shape)
question_info.to_csv('../feat/question_all_avg_embedding.csv', index=False)

del question_info, title_sw_avg_embed_df, title_w_avg_embed_df, describle_sw_avg_embed_df, describle_w_avg_embed_df
gc.collect()

### 对根据问题的话题向量、题目描述等进行聚类后，把问题归类，然后计算每一类别的问题 '回答的内容字数_mean',
### '回答收到的点赞数_mean', '回答收到的取赞数_mean',
###'回答收到的评论数_mean', '回答收藏数_mean', '回答收到的感谢数_mean', '回答收到的被举报数_mean',
###'回答收到的没有帮助数_mean', '回答收到的反对数_mean', '是否为优秀回答_mean', '回答是否被推荐_mean',
###'回答是否被收入圆桌_mean'

data = pd.read_csv("../feat/question_all_avg_embed.csv")
print(data.columns)
ans = data["问题id"]
data.drop(["问题id"], axis=1, inplace=True)
KM = MiniBatchKMeans(n_clusters=1000,  batch_size=100000)
y_pre = KM.fit_predict(data.values)
label_df = pd.DataFrame(y_pre, columns=['category'])
ans = pd.concat([ans, label_df], axis=1)
print(ans.shape)
print(ans.head(4))
ans.to_csv("../feat/question_category.csv", index=False)

question_label = pd.read_csv("../feat/question_category.csv")

print(question_label)

answer_info = pd.read_csv("../../data_set/answer_info_0926.txt", header=None, sep='\t')
answer_info.columns = answer_info_columns
print(answer_info.columns, question_label.columns)
answer_info.drop(['回答id',  '用户id', '回答创建时间', '回答内容单字编码',  '回答内容切词编码', '回答是否被收入圆桌'], axis=1)
print("answer info shape:", answer_info.shape)
answer_info = pd.merge(answer_info, question_label, on='问题id', how='left')
print("answer info shape:", answer_info.shape)
cols = gen_feat1_answer_info_cols + gen_feat2_answer_info_cols
print(answer_info.columns)
for col in cols:
    print(col)
    tmp = answer_info[col].groupby(answer_info['category'])
    question_label = question_label.join(tmp.mean(), on='category')
    question_label = question_label.rename(columns={col: col+'_mean'})

print(question_label.shape)
print(question_label.columns)
print(question_label.head(5))
question_label.to_csv("../feat/question_feat.csv", index=False)