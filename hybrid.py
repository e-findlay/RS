import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import string
import time
import torch
import torch.nn as nn
from nltk.stem.snowball import SnowballStemmer



# Explainable Autoencoder for collaborative filtering
class EAutoRec(nn.Module):
    def __init__(self, user_dim, ex_dim):
        super(EAutoRec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(user_dim + ex_dim, 50),
        )

        self.decoder = nn.Sequential(
            nn.Linear(50, user_dim)
        )
    def forward(self, x, y):
        # concatenate user's rating vector and explainability vector
        h = torch.cat((x,y))
        return self.decoder(torch.sigmoid(self.encoder(h)))




def preprocess_text(text):
    stemmer = SnowballStemmer('english')
    text = [c.lower() for c in text if c not in string.punctuation]
    text = ''.join([stemmer.stem(word) for word in text])
    return text

# load users, businesses dataframes and ratings matrix
users = pd.read_csv('./data/user_covid_data.csv')
businesses = pd.read_csv('./data/business_covid_data.csv')
ratings = np.load('./data/cf_data.npy', allow_pickle=True)
ratings = ratings.tolist()
businesses['processed_text'] = businesses['text'].apply(preprocess_text)

tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
X = tfidf.fit_transform(businesses['processed_text'])



def get_user_profile(user_id, ratings):
    # get index of userid in users dataframe
    user_idx = users[users['user_id'] == user_id].index.to_list()
    user_idx = user_idx[0]
    # find user's average rating
    average_rating = users.iloc[user_idx].average_stars/5
    user_ratings = ratings[user_idx][0]
    profile_vector = []
    total_rating = 0
    # subtract average rating from user's rating and multiply this by the tfidf vector for rated item
    for i in range(len(user_ratings)):
       if user_ratings[i] != 0:
            rating = user_ratings[i] - average_rating
            total_rating += rating
            business_vector = X[i]
            business_vector = business_vector * rating
            profile_vector.append(business_vector)
    if profile_vector == []:
        return np.zeros((1,19186))
    else:
        # normalise vectors by total of users normalised ratings
        for vector in profile_vector:
            vector = vector / total_rating
        return np.sum(np.array(profile_vector))


def find_most_similar_restaurants(profile_vector, k=20, included = False, delivery_only = False, idx=0):
    similarities = []
    # calculate cosine similarity between user profile vector and all restaurant vectors
    for i in range(4055):
        business_vector = X[i]
        similarity = cosine_similarity(profile_vector, business_vector)
        similarities.append(similarity[0][0])
    similarities = np.array(similarities)
    if included:
        # return k items with highest similarity to user profile vector which also offer delivery or Grubhub
        if delivery_only:
            user_rating = ratings[idx][0]
            top_k = []
            sorted_similarities = np.argsort(similarities).tolist()[::-1]
            for i in range(len(sorted_similarities)):
                idx = sorted_similarities[i]
                if (businesses.at[idx, 'Grubhub enabled'] == True or businesses.at[idx, 'delivery or takeout'] == True):
                    top_k.append(idx)
                    if len(top_k) == k:
                        break
        else:
            # return k items with highest similarity to user profile vector
            top_k = np.argsort(similarities)[-k:]
            top_k = top_k.tolist()
    else:
        # return k items with highest similarity to user profile vector which also offer delivery or Grubhub and have not already been rated
        if delivery_only:
            user_rating = ratings[idx][0]
            top_k = []
            sorted_similarities = np.argsort(similarities).tolist()[::-1]
            for i in range(len(sorted_similarities)):
                idx = sorted_similarities[i]
                if user_rating[idx] == 0 and (businesses.at[idx, 'Grubhub enabled'] == True or businesses.at[idx, 'delivery or takeout'] == True):
                    top_k.append(idx)
                    if len(top_k) == k:
                        break
        else:
            # return k items with highest similarity to user profile vector which have not already been rated
            user_rating = ratings[idx][0]
            top_k = []
            sorted_similarities = np.argsort(similarities).tolist()[::-1]
            for i in range(len(sorted_similarities)):
                idx = sorted_similarities[i]
                if user_rating[idx] == 0:
                    top_k.append(idx)
                    if len(top_k) == k:
                        break
    assert(len(top_k) == k)
    return top_k, similarities[top_k]
    

class UI():
    def __init__(self):

        self.user_id = None
        self.permission = False
        self.ratings = None
        self.explain = None
        self.user_idx = None
        self.set_included = None
        self.include = False
        self.k = None
        self.recommended = False
        self.restaurants = None
        self.covid_data = None
        self.take_out = None
        self.run()

    def login(self, user_id):
        self.user_id = user_id
        # find index of user_id in users dataframe
        try:
            self.user_idx = users[users['user_id'] == user_id].index.tolist()[0]
            print("Welcome {}!".format(users['name'][self.user_idx]))
            print('\n')
            # get ratings and explainability vector for user
            self.ratings = ratings[self.user_idx][0]
            self.explain = ratings[self.user_idx][1]
        except:
            print('Login Failed! Username was incorrect!')
            self.user_id = None
            self.ratings = None
            self.user_idx = None


    def logout(self):
        # reset for next user
        self.user_id = None
        self.permission = False
        self.ratings = None
        self.user_idx = None
        self.set_included = None
        self.include = False
        self.k = None
        self.recommended = False
        self.explain = None
        self.restaurants = None
        self.covid_data = None
        self.take_out = None
        print("Thank you for using this service!")
        print('\n')
    
    def run(self):
        while True:
            # get username
            if not self.user_id:
                user_name = input("Please enter your username: ")
                print('\n')
                self.login(user_name)
            else:
                # ask user for permission to use their data
                if not self.permission:
                    print("Please can we use your data for recommendations?")
                    permission = input("We would like to access data on your ratings, reviews and friends. (y/n): ")
                    print('\n')
                    if permission.lower() != 'y':
                        print("We cannot provide recommendations if you do not give us permission!")
                    else:
                        self.permission = True
                else:
                    # set maximum number of recommendations to be displayed to the user
                    if not self.k:
                        k = input("Please enter a maximum number of recommendations (up to 20): ")
                        try:
                            k = int(k)
                            while k > 20 or k < 1:
                                k = input("Please enter a number between 1 and 20: ")
                                k = int(k)
                            self.k = k
                        except:
                            print("Error! Invalid input!")
                        print('\n')
                    # set whether to include items already rated by the user
                    if not self.set_included:
                        included = input("Would you like to include your existing ratings in recommendations? (y/n): ")
                        print('\n')
                        if included.lower() == 'y':
                            self.set_included = True
                            self.include = True
                        elif included.lower() == 'n':
                            self.set_included = True
                            self.include == False
                        else:
                            print('Invalid Input!')
                    if not self.covid_data:
                        # check if user would like to view only takeout/Grubhub options due to covid-19 closures
                        print("Restaurants are currently closed due to Covid-19.")
                        only_takeout = input("Would you like to only see restaurants offering takeout or Grubhub options? (y/n): ")
                        print('\n')
                        if only_takeout.lower() == 'y':
                            self.covid_data = True
                            self.take_out = True
                        elif only_takeout.lower() == 'n':
                            self.covid_data = True
                            self.take_out = False
                        else:
                            print("Invalid Input!")
                    # generate recommendations
                    if not self.recommended:
                        self.restaurants = self.recommend(self.user_id, self.k, self.include)
                        self.recommended = True
                        print('\n')
                        # allow user to add a rating to recommendations
                        add_rating = input('Would you like to rate a restaurant? (y/n) ')
                        print('\n')
                        if add_rating.lower() == 'y':
                            valid_input = False
                            while not valid_input:
                                restaurant = input('Please enter the number next to the restaurant you would like to rate: ')
                                print('\n')
                                try:
                                    restaurant = int(restaurant)
                                    print(restaurant)
                                    restaurant_name = self.restaurants[restaurant]
                                    valid_input = True
                                except:
                                    print("Error: The number you entered was not in your recommendations")
                            rating_provided = False
                            while not rating_provided:
                                rating = input('Please enter a rating for {} between 1 and 5: '.format(restaurant_name))
                                print('\n')
                                try:
                                    rating = round(float(rating),1)
                                    assert(1 <= rating <= 5)
                                    rating_provided = True
                                except:
                                    print("Please enter a number between 1 and 5")
                            b_id = businesses[businesses['name'] == restaurant_name].index.tolist()[0]
                            # update user's ratings
                            ratings[self.user_idx][0][b_id] = rating
                            self.ratings = ratings[self.user_idx][0]
                            print('\n')
                            self.restaurants = self.recommend(self.user_id, self.k, self.include)
                            self.recommended = True
                            print('\n')


                    else:
                        print('Press L to logout or R to see your recommendations again')
                        log_input = input('')
                        if log_input.lower() == 'l':
                            self.logout()
                        elif log_input.lower() == 'r':
                            self.recommended = False
                        else:
                            print('Invalid Command!')


    
    def recommend(self, user_id, k=20, include=True):
        # calculate user profile
        profile_vector = get_user_profile(user_id, ratings)
        # find k most similar restaurants using recommender 1
        top_k, scores = find_most_similar_restaurants(profile_vector, k, include, self.take_out, self.user_idx)
        business_names = []
        max_len = 0
        for i in top_k:
            if len(businesses['name'][i]) > max_len:
                max_len = len(businesses['name'][i])
            business_names.append(businesses['name'][i])
        business_ratings = []
        # predict ratings for user using recommender 2
        model_params = torch.load('./model.pt')
        model = EAutoRec(4055, 4055)
        # load autoencoder parameters
        model.load_state_dict(model_params['model'])
        pred = model(torch.tensor(self.ratings), torch.tensor(self.explain))
        pred = pred.detach().numpy()
        # filter ratings based on items from recommender 1
        pred = pred[top_k]
        explain = np.array(self.explain)[top_k]
        explain = explain.tolist()
        collab_scores = pred.tolist()
        prev_score = -1
        for i in range(len(scores)):
            score = round(scores[i],1)
            if score == prev_score:
                prev_score = score
                scores[i] = collab_scores[i]
            prev_score = score
        business_scores = zip(scores, business_names, explain)
        business_scores = sorted(business_scores, reverse=True)
        similar_restaurants = []
        friends_restaurants = []
        # filter out 
        # low scoring items
        for score, restaurant, explain in business_scores:
            if score > 0.6:
                if explain > 0:
                    score = min(round(score * 5,1),5.0)
                    friends_restaurants.append((restaurant, score))
                else:
                    score = min(round(score * 5,1),5.0)
                    similar_restaurants.append((restaurant, score))
        counter = 0
        restaurants = {}
        if len(friends_restaurants) > 0:
            print('\n')
            print("These restaurants are similar to ones your friends liked: ")
            print('\tRestaurant{}Predicted Rating'.format((max_len - len('Restaurant')) * ' '))
        for i in friends_restaurants:
            counter += 1
            restaurants[counter] = i[0]
            print('{}{}{}{}{}'.format(counter, (5 - (counter // 10)) * ' ', i[0], (max_len - len(i[0]) + 4) * ' ', i[1]))
        if len(similar_restaurants) > 0:
            print('\n')
            print("These restaurants are similar to ones that you liked: ")
            print('\tRestaurant{}Predicted Rating'.format((max_len - len('Restaurant')) * ' '))
        for i in similar_restaurants:
            counter += 1
            restaurants[counter] = i[0]
            print('{}{}{}{}{}'.format(counter, (5 - (counter // 10)) * ' ', i[0], (max_len - len(i[0]) + 4) * ' ', i[1]))
        return restaurants



ui = UI()
ui.run()

                    
