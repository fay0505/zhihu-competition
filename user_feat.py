import pandas  as pd
import gc
import numpy as np
from sklearn.decomposition import PCA




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


#### 计算用户关注话题、感兴趣话题、最感兴趣话题的平均embed、以及问题绑定的话题的平均embed

topic_vectors = pd.read_csv('../../data_set/topic_vectors_64d.txt', names=['话题id', 'embed'], sep='\t')
topic_vectors['embed'] = topic_vectors['embed'].apply(parse_str)
print(topic_vectors.head(2))


topic_ids = topic_vectors['话题id'].tolist()
topic_vectors.drop(['话题id'], axis=1, inplace=True)
topic_vectors = topic_vectors.T
topic_vectors.columns = topic_ids

print("topic vectors shape:", topic_vectors.shape)
print(topic_vectors.head(5))


user_info = pd.read_csv('../../data_set/member_info_0926.txt', header=None, sep='\t')
user_info.columns = ['用户id','性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问频率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值','关注话题','感兴趣话题']
user_info  = user_info.drop(['性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问频率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值'], axis=1)
print("user info shape:",user_info.shape)
print(user_info.head(5))

user_info['感兴趣话题'] = user_info['感兴趣话题'].apply(parse_map)
user_info['最感兴趣话题'] = user_info['感兴趣话题'].apply(most_interest_topic)
user_info['感兴趣话题'] = user_info['感兴趣话题'].apply(get_interest_topics)
user_info['关注话题'] = user_info['关注话题'].apply(parse_list_1)

print("解析完user info ", user_info['感兴趣话题'].head(5))
print("解析完user info ", user_info['最感兴趣话题'].head(5))
print("解析完user info ", user_info['关注话题'].head(5))

attend_topic_avg_embedding = []
interest_topic_avg_embedding = []
most_interest_topic_avg_embedding = []

for index, row in user_info.iterrows():
    print(index)
    attend_topic_avg_embedding.append(cal_avg_embedding(row['关注话题']))
    interest_topic_avg_embedding.append(cal_avg_embedding(row['感兴趣话题']))
    most_interest_topic_avg_embedding.append(cal_avg_embedding(row['最感兴趣话题']))

print("save attend_topic_avg_embedding_df")
attend_topic_avg_embedding_df = pd.DataFrame(attend_topic_avg_embedding, columns=["vec{0}".format(i) for i in range(64)])
attend_topic_avg_embedding_df['用户id'] = user_info['用户id']
attend_topic_avg_embedding_df.to_csv("../feat/user_attend_topic_avg_embedding.csv", index=False)
del attend_topic_avg_embedding_df, attend_topic_avg_embedding
gc.collect()

print("save interest_topic_avg_embedding_df")
interest_topic_avg_embedding_df = pd.DataFrame(interest_topic_avg_embedding, columns=["vec{0}".format(i) for i in range(64)])
interest_topic_avg_embedding_df['用户id'] = user_info['用户id']
interest_topic_avg_embedding_df.to_csv("../feat/user_interest_topic_avg_embedding.csv", index=False)
del interest_topic_avg_embedding_df, interest_topic_avg_embedding
gc.collect()
print("save most_interest_topic_avg_embedding_df")
most_interest_topic_avg_embedding_df = pd.DataFrame(most_interest_topic_avg_embedding, columns=["vec{0}".format(i) for i in range(64)])
most_interest_topic_avg_embedding_df['用户id'] = user_info['用户id']
most_interest_topic_avg_embedding_df.to_csv("../feat/user_most_interest_topic_avg_embedding.csv", index=False)
del most_interest_topic_avg_embedding_df, most_interest_topic_avg_embedding
gc.collect()


'''
求 用户关注话题、感兴趣话题、最感兴趣话题和 题目的切词、绑定话题 向量cos距离
'''
#### 以下是生成embed 向量之间的距离特征代码
question_tw_feats = ["t_w_vec{0}".format(i) for i in range(64)]
question_topic_feats = ["topic_vec{0}".format(i) for i in range(64)]

question_embed = pd.read_csv('../feat/question_all_avg_embed.csv')
question_ids = question_embed['问题id'].tolist()

question_tw_embed = question_embed[question_tw_feats]
question_tw_embed = question_tw_embed.T
question_tw_embed.columns = question_ids

question_topic_embed = question_embed[question_topic_feats]
question_topic_embed = question_topic_embed.T
question_topic_embed.columns = question_ids

print("处理之后的question embed...", question_tw_embed.shape, question_topic_embed.shape)


user_attend_topic_avg_embed = pd.read_csv('../feat/user_attend_topic_avg_embedding.csv')
print("处理之前的attend_topic_avg_embed", user_attend_topic_avg_embed.head(5))
user_ids = user_attend_topic_avg_embed['用户id'].tolist()
user_attend_topic_avg_embed.drop(['用户id'], axis=1, inplace=True)
user_attend_topic_avg_embed = user_attend_topic_avg_embed.T
user_attend_topic_avg_embed.columns = user_ids
print("处理之后的attend_topic_avg_embed", user_attend_topic_avg_embed.head(5))

user_interest_topic_avg_embed = pd.read_csv("../feat/user_interest_topic_avg_embedding.csv")
print("处理之前的user_interest_topic_avg_embed", user_interest_topic_avg_embed.head(5))
user_ids = user_interest_topic_avg_embed['用户id'].tolist()
user_interest_topic_avg_embed.drop(['用户id'], axis=1, inplace=True)
user_interest_topic_avg_embed = user_interest_topic_avg_embed.T
user_interest_topic_avg_embed.columns = user_ids
print("处理之后的attend_topic_avg_embed", user_interest_topic_avg_embed.head(5))


user_most_interest_topic_avg_embed = pd.read_csv("../feat/user_most_interest_topic_avg_embedding.csv")
print("处理之前的user_most_interest_topic_avg_embed", user_most_interest_topic_avg_embed.head(5))
user_ids = user_most_interest_topic_avg_embed['用户id'].tolist()
user_most_interest_topic_avg_embed.drop(['用户id'], axis=1, inplace=True)
user_most_interest_topic_avg_embed = user_most_interest_topic_avg_embed.T
user_most_interest_topic_avg_embed.columns = user_ids
print("处理之后的attend_topic_avg_embed", user_most_interest_topic_avg_embed.head(5))


def vector_cos(data1, data2):
    num = np.dot(data1, data2)
    denom = np.linalg.norm(data1) * np.linalg.norm(data2)
    cos = num / (denom + 1e-4)

    return 0.5 + 0.5*cos



def cal_dis_feat(data, name="train"):
    print("计算各种距离的特征..")
    print(data.head(5))

    # 用户最感兴趣话题的平均embed分别和问题的绑定话题、题目切词的平均embed 的cos
    most_interest_topic_question_tw_cos = []
    attend_topic_question_tw_cos = []
    interest_topic_question_tw_cos = []

    most_interest_topic_question_cos = []
    attend_topic_question_cos = []
    interest_topic_question_cos = []

    total = data.shape[0]
    for index, row in data.iterrows():

        user_id = row['用户id']
        question_id = row['问题id']
        interest_topic_question_tw_cos.append(vector_cos(question_tw_embed[question_id].values, user_interest_topic_avg_embed[user_id].values))
        most_interest_topic_question_tw_cos.append(vector_cos(question_tw_embed[question_id].values, user_most_interest_topic_avg_embed[user_id].values))
        attend_topic_question_tw_cos.append(vector_cos(question_tw_embed[question_id].values, user_attend_topic_avg_embed[user_id].values))

        interest_topic_question_cos.append(vector_cos(question_topic_embed[question_id].values, user_interest_topic_avg_embed[user_id].values))
        most_interest_topic_question_cos.append(vector_cos(question_topic_embed[question_id].values, user_most_interest_topic_avg_embed[user_id].values))
        attend_topic_question_cos.append(vector_cos(question_topic_embed[question_id].values, user_attend_topic_avg_embed[user_id].values))

        print("index:{0}, finish:{1} %".format(index, (index + 1) * 100 / total))

    print("{0} feat 计算完毕".format(name))
    print(data.shape)
    data['interest_topic_question_tw_cos'] = interest_topic_question_tw_cos
    data['most_interest_topic_question_tw_cos'] = most_interest_topic_question_tw_cos
    data['attend_topic_question_tw_cos'] = attend_topic_question_tw_cos

    data['interest_topic_question_cos'] = interest_topic_question_cos
    data['most_interest_topic_question_cos'] = most_interest_topic_question_cos
    data['attend_topic_question_cos'] = attend_topic_question_cos

    data.to_csv("../feat/{0}_cos_feat.csv".format(name), index=False)


train =  pd.read_csv('../../data_set/invite_info_0926.txt', header=None, sep='\t')
train.columns = ['问题id', '用户id', '邀请创建时间','是否回答']

cal_dis_feat(train)
del train
test = pd.read_csv('../../data_set/invite_info_evaluate_2_0926.txt', header=None, sep='\t')
test.columns = ['问题id', '用户id', '邀请创建时间']
cal_dis_feat(test, 'test1')


