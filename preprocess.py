import pandas as pd
import os
import json
import csv
import numpy as np


#DATA Preprocessing

chunk_size=100000

infile = './data/yelp_academic_dataset_business.csv'
in_review_file = './data/yelp_academic_dataset_review.csv'
in_user_file = './data/yelp_academic_dataset_user.csv'
in_covid_file = './data/yelp_academic_dataset_covid.csv'

df_business = pd.read_csv(infile)
df_reviews = pd.read_csv(in_review_file, chunksize=100000)
df_users = pd.read_csv(in_user_file)
df_covid = pd.read_csv(in_covid_file)


# select businesses in restaurant category
is_restaurant = df_business.categories.str.contains('Restaurants', na=False)
# remove non restaurant businesses from dataset
df_business = df_business[is_restaurant]


# remove closed businesses from dataset
df_business = df_business[df_business['is_open']==1]
# get covid information for businesses
df_business = df_business.merge(df_covid, how='inner', on='business_id')
df_business = df_business[['business_id', 'Grubhub enabled', 'delivery or takeout', 'name', 'stars', 'city', 'state', 'address', 'postal_code']]
b_data = df_business['business_id']

chunks = []
for chunk in df_reviews:
    # avoid name conflict for stars when merging with df_business
    chunk = chunk.rename(columns={'stars': 'review_stars'})
    # merge chunk with business df
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True, join='outer', axis=0)
# 1 month timeframe
df = df[df['date'] >= '2019-12-01']
# drop business reviews with no text
#df = df.dropna(subset=['text'])
df = df.merge(b_data, how='inner', on='business_id')
print(len(df))
print(len(df_users))
# drop users with less than 20 reviews
df_users = df_users[df_users['review_count'] >= 20]
print(len(df_users))
# drop users with no friends
df_users = df_users.dropna(subset=['friends'])
print(len(df_users))
print(df_users.columns)
df_users = df_users.merge(df, how='inner', on='user_id')
df_users = df_users[['user_id','friends', 'average_stars', 'name', 'review_count']]
df_users.drop_duplicates()
print(len(df_users))
print(df_users.columns)
friend_set = set(df_users['user_id'])
def friends_to_list(friends):
    lst = friends.split(', ')
    f = []
    for friend in lst:
        if friend in friend_set:
            f.append(friend)
    if len(f) >= 1:
        return f
    else:
        return []

friend_set = set(df_users['user_id'])
df_users['friends_list'] = df_users['friends'].apply(friends_to_list)
df_users = df_users[df_users['friends_list'].map(lambda lst: len(lst) > 0)]
print(len(df_users))
df = df.merge(df_users, how='inner', on='user_id')
b_id = df[['business_id', 'text']]
# concatenate reviews for businesses
b_id = b_id.groupby(['business_id'], as_index=False)['text'].apply(', '.join)
print(b_id.head())
print(len(df))
b_id = b_id.drop_duplicates()
print(len(df_business))
df_business = df_business.merge(b_id, how='inner', on='business_id')
print(len(df_business))
df_b = df_business['business_id']
df = df.merge(df_b, on='business_id', how='inner')
df_u = df['user_id']
df_users = df_users.merge(df_u, on='user_id', how='inner')
print(df_users.columns)
# find users who have rated businesses reviewed within the timeframe
df_users = df_users.drop(['friends_list'], axis=1)
df_users = df_users.drop_duplicates()
friend_set = set(df_users['user_id'])
df_users['friends_list'] = df_users['friends'].apply(friends_to_list)
print(len(df_users), len(set(df_users['user_id'])), len(df), len(df_business))

df_users.reset_index(drop=True, inplace=True)
df_business.reset_index(drop=True, inplace=True)

df_users['index'] = df_users.index
df_business['index'] = df_business.index

ratings = []
users_ids = df_users['index'].to_list()
u_names = df_users['user_id'].to_list()
b_names = df_business['business_id'].to_list()

business_ids = df_business['index'].to_list()
u_to_int = {}
b_to_int = {}
for i in range(len(u_names)):
    u_to_int[u_names[i]] = users_ids[i]
for i in range(len(b_names)):
    b_to_int[b_names[i]] = business_ids[i]

# create rating matrix
for u in users_ids:
    u_ratings = []
    for b in business_ids:
        u_ratings.append(0)
    ratings.append(u_ratings)

# fill ratings matrix with ratings
for i, row in df.iterrows():
    u_id = row.user_id
    if u_id in u_names:
        b_id = row.business_id
        b_id = b_to_int[b_id]
        u_id = u_to_int[u_id]
        r = (row.review_stars) / 5
        ratings[u_id][b_id] = r

# if user's friend is in users dataframe, find friend's index else drop friend from friends list
def friends_to_idx(friends):
    lst = []
    for f in friends:
        try:
            idx = df_users[df_users['user_id'] == f]['index']
            lst.append(idx)
        except:
            continue
    return_lst = []
    for elem in lst:
        return_lst.extend(elem)
    return return_lst

df_users['friend_idx'] = df_users['friends_list'].apply(friends_to_idx)
print(len(df_users))
print(df_users.head())
print(len(ratings))



def generate_explain_vector(friends, n):
    explainability_vector = []
    # find maximum rating and sum of all ratings among friends for each item
    for i in range(len(friends[0])):
        maximum = 0
        total = 0
        for j in range(n):
            total += friends[j][i]
            if friends[j][i] > maximum:
                maximum = friends[j][i]
        # if friends rated an item divide sum by maximum * number of friends else fill in position with 0
        if maximum > 0:
            result = total / (maximum * n)
            explainability_vector.append(result)
        else:
            explainability_vector.append(0)
    return explainability_vector


def get_user_explainability_vector(friends):
    friend_vectors = []
    n = len(friends)
    # get friends' ratings
    for f in friends:
        f_vector = ratings[f]
        friend_vectors.append(f_vector)
    # if user has no friends set all to 0
    if friend_vectors == []:
        friend_vectors.append([0] * len(df_business))
        n = 1
    # calculate explainability vector from friends' ratings
    explainability_vector = generate_explain_vector(friend_vectors, n)
    assert(len(explainability_vector) == len(df_business))
    return explainability_vector

df_users['friend_vectors'] = df_users['friend_idx'].apply(get_user_explainability_vector)

train_data = []
c = 0
for i in range(len(ratings)):
    friends = df_users['friend_vectors'][i]
    rating = ratings[i]
    if rating == [0] * len(df_business):
        c += 1
    data = []
    data.append(rating)
    data.append(friends)
    train_data.append(data)
print(c)
print(np.array(train_data).shape)
np.save('./data/cf_data.npy', train_data)

df_users.to_csv('./data/user_covid_data.csv', index=False)
df_business.to_csv('./data/business_covid_data.csv', index=False)
print('done')