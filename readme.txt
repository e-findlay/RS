To run the recommender system run the command python hybrid.py.
This file requires that the following packages are installed:
    pytorch
    sklearn
    numpy
    nltk

To access the recommendation a user_id is required to login. Any user_id from the user_covid_data.csv file should work such as:
dIIKEfOgo0KqUfGQvGikPg
AG_bM2ATIvqgcGTBX-gw2w

If you want to run the data preprocessing file, place the original business, user, review and covid files from the Yelp Dataset in 
the data directory and run python preprocess.py. However the data has already been preprocessed and the required 
files are in the data directory.

The cf.py file contains code for training the autoencoder model. However this model has been pretrained and is 
saved as model.pt in the data directory.