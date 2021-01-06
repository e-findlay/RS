import pandas as pd
import numpy as np
in_user_file = 'C:/Users/fox2e/RS/data/yelp_academic_dataset_user.csv'
df_users = pd.read_csv(in_user_file)
print(len(df_users))
df_users = df_users[['user_id', 'friends', 'average_stars', 'review_count']]
print(df_users.head())