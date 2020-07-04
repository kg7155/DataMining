import random
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

''' FUNCTIONS
'''
# predict using MF and biases
def predict(u, a, P, Q, b, b_u, b_a):
    if u not in b_u:
        b_user = 0
    else:
        b_user = b_u[u]
    
    if a not in b_a:
        b_artist = 0
    else:
        b_artist = b_a[a]
        
    pred = np.dot(P[u], Q[a]) + b + b_user + b_artist
    
    if pred < 0:
        return 0
    return pred

# predict using only biases (mean of user's and artist's mean)
def predict_biases(u, a, test):
    if u not in b_u:
        b_user = 0
    else:
        b_user = test.loc[test["userID"] == u, "user_mean"].iloc[0]

    if a not in b_a:
        b_artist = 0
    else:
        b_artist = test.loc[test["artistID"] == a, "artist_mean"].iloc[0]
    
    return 1/2 * b_user + 1/2 * b_artist

# calculate RMSE on given test set
def compute_rmse(test, P, Q, b, b_u, b_a):
    rmse = 0
    for index, row in test.iterrows():
        u = row["userID"]
        a = row["artistID"]
        r = row["weight"]
        r_pred = predict(u, a, P, Q, b, b_u, b_a)
        #r_pred = predict_biases(u, a, test)
        rmse += (r-r_pred)**2

    return math.sqrt(rmse/len(test))

''' MAIN
'''
if __name__ == "__main__":
    # load data
    train_ua_path = "data/user_artists_training.dat"
    #train_ua_path = "data/user_artists_training_my_ratings.dat"
    test_ua_path = "data/user_artists_test.dat"
    
    # create dataframes out of given train and test set
    train_ua = pd.read_csv(train_ua_path, sep="\t", header=0)
    test_ua = pd.read_csv(test_ua_path, sep="\t", header=0)
    
    # count distinct users and artists
    num_users = len(train_ua["userID"].value_counts())
    num_artists = len(train_ua["artistID"].value_counts())
    
    ''' PARAMS '''
    K = 50
    reg_fact = 1000
    alpha = 0.000025
    beta = .05

    # prepare P and Q matrices (dicts in my case)
    P = {}
    for user in train_ua["userID"].unique():
        P[user] = np.array([random.uniform(0.1, 1) for i in range(K)])

    Q = {}
    for artist in train_ua["artistID"].unique():
        Q[artist] = np.array([random.uniform(0.1, 1) for i in range(K)])

    # initialize the biases: user, artist, global
    b_u = {}
    for user in train_ua["userID"].unique():
        b_u[user] = 0

    b_a = {}
    for artist in train_ua["artistID"].unique():
        b_a[artist] = 0

    b = train_ua["weight"].mean()
    
    # calculate user and artist mean and add to dataframe as columns
    train_ua["user_mean"] = train_ua["userID"].map(train_ua.groupby(["userID"])["weight"].mean())
    train_ua["artist_mean"] = train_ua["artistID"].map(train_ua.groupby(["artistID"])["weight"].mean())

    # split train set into train and test set
    my_train, my_test = train_test_split(train_ua, test_size=0.30, random_state=42)
    
    prev_error = np.inf
    error = compute_rmse(my_test, P, Q, b, b_u, b_a)
    
    # main matrix factorization algorithm
    while (error < prev_error - 1e-4):
        prev_error = error
        for index, row in my_train.iterrows():
            u = row["userID"]
            a = row["artistID"]
            r = row["weight"]
            r_approx_ui = predict(u, a, P, Q, b, b_u, b_a)
            e_ui = r - r_approx_ui

            # update biases
            b_u[u] += beta * (e_ui-b_u[u])
            b_a[a] += beta * (e_ui-b_a[a])
           
            # update dicts
            P[u] = P[u] + alpha * (e_ui * Q[a] - reg_fact * P[u])
            Q[a] = Q[a] + alpha * (e_ui * P[u] - reg_fact * Q[a])

        error = compute_rmse(my_test, P, Q, b, b_u, b_a)
        print("RMSE: ", error)
   
    # predict using homework's test data
    with open("predictions.txt", "w") as f:
        for index, row in test_ua.iterrows():
            u = row["userID"]
            a = row["artistID"]

            # for new user predict an average rating, for new artist global mean
            if u not in P:
                r_pred = train_ua.loc[train_ua["artistID"] == a, "artist_mean"].iloc[0]
            elif a not in Q:
                r_pred = train_ua["weight"].mean()
            else:
                r_pred = predict(u, a, P, Q, b, b_u, b_a)

            #r_pred = predict_biases(u, a, train_ua)
            
            print(r_pred, file=f)
    
    '''
    # Third part of assignment
    artists_path = "data/artists.dat"
    artists = pd.read_csv(artists_path, sep="\t", header=0)

    ratings = {}
    for artist in train_ua["artistID"].unique():
        r_pred = predict(9999, int(artist), P, Q, b, b_u, b_a)
        ratings[int(artist)] = r_pred

    # sort dict in descending order
    sorted_ratings = sorted(ratings, key=ratings.get, reverse=True)

    # print first 10 values
    for i in range (10):
        id = sorted_ratings[i]
        name = artists.loc[artists["id"] == id, "name"].iloc[0]
        print(id, ratings[id], name)
    '''
    print("Done")