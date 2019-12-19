import pandas as pd
import gc
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import gc
from  sklearn.metrics import calinski_harabasz_score

def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: str(x), str(d).split(',')))


def parse_map(d):
    if d == '-1':
        return {}
    return dict([z.split(':')[0], z.split(':')[1]] for z in d.split(','))


def less_interest_topic(d):
    if len(d) == 0:
        return [0]
    return [list(d.keys())[np.argmin(list(d.values()))]]


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

less_interest_avg_embedding_df = pd.read_csv("../feat/user_less_interest_topic_avg_embedding.csv")

attend_topic_avg_embedding_df = pd.read_csv("../feat/user_attend_topic_avg_embedding.csv")
attend_topic_avg_embedding_df.drop(['用户id'], inplace=True,  axis=1)
attend_topic_avg_embedding_df.columns = ["attend_topic_vec{0}".format(i) for i in range(64)]

interest_topic_avg_embedding_df = pd.read_csv("../feat/user_interest_topic_avg_embedding.csv")
interest_topic_avg_embedding_df.drop(['用户id'], inplace=True,  axis=1)
interest_topic_avg_embedding_df.columns = ["interest_topic_vec{0}".format(i) for i in range(64)]

most_interest_topic_avg_embedding_df  = pd.read_csv("../feat/user_most_interest_topic_avg_embedding.csv")
most_interest_topic_avg_embedding_df.drop(['用户id'], inplace=True,  axis=1)
most_interest_topic_avg_embedding_df.columns =  ["most_interest_topic_vec{0}".format(i) for i in range(64)]

print(less_interest_avg_embedding_df.shape)
user_info = pd.concat([less_interest_avg_embedding_df, attend_topic_avg_embedding_df], axis=1)
user_info = pd.concat([user_info, attend_topic_avg_embedding_df],  axis=1)
user_info = pd.concat([user_info, interest_topic_avg_embedding_df],  axis=1)
user_info = pd.concat([user_info, most_interest_topic_avg_embedding_df],  axis=1)
print(user_info.shape)
user_info.to_csv("../feat/user_all_embed.csv", index=False)

ans = user_info['用户id']
user_info.drop(['用户id'], axis=1, inplace=True)

best_n = 200
max_scores = 0
for n in [200, 500, 800, 1000]:
    KM = MiniBatchKMeans(n_clusters=n, random_state=2019, batch_size=15000)
    y_pre = KM.fit_predict(user_info.values)
    #print(y_pre)
    score = calinski_harabasz_score(user_info.values, y_pre)
    print(n, score)
    if score > max_scores:
        best_n = n
        max_scores = score

KM = MiniBatchKMeans(n_clusters=best_n, random_state=2019, batch_size=15000)
y_pre = KM.fit_predict(user_info.values)
label_df = pd.DataFrame(y_pre, columns=['category'])
ans = pd.concat([ans, label_df], axis=1)
print(ans.shape)
print(ans.head(4))
ans.to_csv("../feat/user_category.csv", index=False)
exit()

topic_vectors = pd.read_csv('../../data_set/topic_vectors_64d.txt', names=['话题id', 'embed'], sep='\t')
topic_vectors['embed'] = topic_vectors['embed'].apply(parse_str)
print(topic_vectors.head(2))


topic_ids = topic_vectors['话题id'].tolist()
topic_vectors.drop(['话题id'], axis=1, inplace=True)
topic_vectors = topic_vectors.T
topic_vectors.columns = topic_ids

print("topic vectors shape:", topic_vectors.shape)
print(topic_vectors.head(5))
#print(topic_vectors['T1'])

user_info = pd.read_csv('../../data_set/member_info_0926.txt', header=None, sep='\t')
user_info.columns = ['用户id','性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问频率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值','关注话题','感兴趣话题']
user_info  = user_info.drop(['性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问频率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值'], axis=1)
print("user info shape:",user_info.shape)
print(user_info.head(5))

user_info['感兴趣话题'] = user_info['感兴趣话题'].apply(parse_map)
user_info['最小感兴趣话题'] = user_info['感兴趣话题'].apply(less_interest_topic)


print("解析完user info ", user_info['最小感兴趣话题'].head(5))


less_interest_topic_avg_embedding = []

for index, row in user_info.iterrows():
    print(index)

    less_interest_topic_avg_embedding.append(cal_avg_embedding(topic_vectors, row['最小感兴趣话题']))

print("save attend_topic_avg_embedding_df")
less_interest_avg_embedding_df = pd.DataFrame(less_interest_topic_avg_embedding, columns=["less_interest_topic_vec{0}".format(i) for i in range(64)])
less_interest_avg_embedding_df['用户id'] = user_info['用户id']
less_interest_avg_embedding_df.to_csv("../feat/user_less_interest_topic_avg_embedding.csv", index=False)
del less_interest_topic_avg_embedding
gc.collect()

