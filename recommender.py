import pandas as pd
import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import csv
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


#DATA Preprocessing
'''
infile = 'C:/Users/fox2e/RS/yelp_dataset/yelp_academic_dataset_business.json'
in_review_file = 'C:/Users/fox2e/RS/yelp_dataset/yelp_academic_dataset_review.json'
in_tip_file = 'C:/Users/fox2e/RS/yelp_dataset/yelp_academic_dataset_tip.json'
in_user_file = 'C:/Users/fox2e/RS/yelp_dataset/yelp_academic_dataset_user.json'
in_checkin_file = 'C:/Users/fox2e/RS/yelp_dataset/yelp_academic_dataset_checkin.json'
in_covid_file = 'C:/Users/fox2e/RS/covid_19_dataset/covid_19_dataset_2020_06_10/yelp_academic_dataset_covid_features.json'
outfile = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_business.csv'
out_review_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_review.csv'
out_tip_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_tip.csv'
out_user_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_user.csv'
out_checkin_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_checkin.csv'
out_covid_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_covid.csv'
chunk_size=100000

df_reviews = pd.read_json(in_review_file, lines=True, dtype = {
    'review_id': str, 'user_id': str, 'business_id': str, 'stars': int, 'date': str,
    'text': str, 'useful': int, 'funny': int, 'cool': int
}, chunksize=chunk_size)

df_business = pd.read_json(infile, lines=True)
df_business.to_csv(outfile, index=False)
df_tip = pd.read_json(in_tip_file, lines=True)
df_tip.to_csv(out_tip_file, index=False)
df_user = pd.read_json(in_user_file, lines=True)
df_user.to_csv(out_user_file, index=False)
df_checkin = pd.read_json(in_checkin_file, lines=True)
df_checkin.to_csv(out_checkin_file, index=False)
df_covid = pd.read_json(in_covid_file, lines=True)
df_covid.to_csv(out_covid_file, index=False)


infile = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_business.csv'
in_review_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_review.csv'
in_tip_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_tip.csv'
in_user_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_user.csv'
in_checkin_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_checkin.csv'
in_covid_file = 'C:/Users/fox2e/RS/data//yelp_academic_dataset_covid.csv'

df_business = pd.read_csv(infile)
df_reviews = pd.read_csv(in_review_file, chunksize=100000)
df_tips = pd.read_csv(in_tip_file)
df_users = pd.read_csv(in_user_file)
df_checkin = pd.read_csv(in_checkin_file)
df_covid = pd.read_csv(in_covid_file)

# select businesses in restaurant category
is_restaurant = df_business.categories.str.contains('Restaurants', na=False)
print(set(is_restaurant))
# remove non restaurant businesses from dataset
df_business = df_business[is_restaurant]
# drop categories column from dataframe
#df_business = df_business.drop(['categories'])


# remove closed businesses from dataset
df_business = df_business[df_business['is_open']==1]

chunks = []
for chunk in df_reviews:
    # avoid name conflict for stars when merging with df_business
    chunk = chunk.rename(columns={'stars': 'review_stars'})
    # merge chunk with business df
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True, join='outer', axis=0)
df_business_reviews = df_business.merge(df, how='inner', on='business_id')

df_business_reviews = df_business_reviews[df_business_reviews.date >= '2019-01-01']

df_business_reviews = df_business_reviews.merge(df_covid, how='inner', on='business_id')
df_business_reviews = df_business_reviews[['business_id', 'name', 'stars', 'city', 'latitude', 'longitude', 'review_count', 'review_stars', 'text', 'user_id']]
df_business_reviews.to_csv('C:/Users/fox2e/RS/data/business_reviews.csv', index=False)
df_users = df_users[['user_id', 'friends', 'average_stars', 'review_count']]
df_users.to_csv('C:/Users/fox2e/RS/data/users.csv', index=False)
'''
'''
df = pd.read_csv('C:/Users/fox2e/RS/data/users.csv')
df_business_reviews = pd.read_csv('C:/Users/fox2e/RS/data/business_reviews.csv')
df = df_business_reviews.merge(df, how='inner', on='user_id')
df.to_csv('C:/Users/fox2e/RS/data/data.csv', index=False)
'''
df = pd.read_csv('C:/Users/fox2e/RS/data/data.csv')
id_map = {}
count = 0
for i in df['user_id']:
    if i not in id_map.keys():
        id_map[i] = count
        count += 1
ids = []
for i in df['user_id']:
    ids.append(id_map[i])
df['user_values'] = ids

bid_map = {}
count = 0
for i in df['business_id']:
    if i not in bid_map.keys():
        bid_map[i] = count
        count += 1
bids = []
for i in df['business_id']:
    bids.append(bid_map[i])
df['business_values'] = bids

item_features = df[['business_id', 'name', 'stars', 'city', 'latitude', 'longitude', 'review_count_x', 'text', 'business_values']]
user_features = df[['user_id', 'average_stars', 'review_count_y', 'review_stars', 'user_values']]

bids = list(set(item_features['business_values']))
reviews = [''] * len(bids)
for i in item_features.itertuples():
    reviews[int(i[7])] += i[8]
data = zip(bids, reviews)
item_reviews = pd.DataFrame(data, columns=['business_values', 'text'])
print(item_reviews.head())
item_reviews['text'] = item_reviews['text'].str.lower().str.split()
#stopwords = stopwords.words('english')
# remove stopwords
#item_reviews['text'] = item_reviews['text'].apply(lambda x: [i for i in x if i not in stopwords])
stemmer = PorterStemmer('english')
# apply stemming
item_reviews['text'] = item_reviews['text'].apply(lambda x: [stemmer.stem(i) for i in x])

text = [i for i in item_reviews['text']]

tfidf = TfidfVectorizer()
item_train_data = tfidf.fit_transform(text)





def similarity(vec1, vec2):
    return cosine_similarity(vec1, vec2)

def knn(user_profile):
    pass

class AutoEncoder(nn.Module):

    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim //8, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.ReLU()
        )
    
    def forward(self, X):
        return self.decoder(self.encoder(X))

class MLP(nn.Module):

    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def forward(self, X):
        return F.sigmoid(self.MLP(X))

def train(n_epochs):
    user_ae = AutoEncoder(3)
    item_ae = AutoEncoder(8)
    for i in range(n_epochs):


'''
##################################################################################################################################


Login function to verify access to user recommendations through CLI
'''
def login(userid):
    return userid in df['user_id']


def cli():
    print('Welcome!')
    logged_in = False
    while not logged_in:
        user_id = input('Please enter your user_id:')
        logged_in = login(user_id)

