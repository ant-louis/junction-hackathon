import os
import time
import datetime
from contextlib import contextmanager
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label, datetime.timedelta(seconds=end-start)))


def load_data(path, to_split=True):
    """
    Load the csv file and returns (X,y).
    """
    # Read the csv file
    df = pd.read_csv(path, header=0, index_col=0)

    # Get the output values
    y = df['stop_passengers'].values.squeeze()

    # Get the input values
    features = ['stop_lat', 'stop_long']
    X = df[features].values.squeeze()
    
    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if to_split:
        return X_train, X_test, y_train, y_test
    else:
        return X, y


def train(path, to_split=True):
    """
    Train the model.
    """
    filename = "_Models/Lin_reg.pkl"
    
    # Load the training (and testing) set(s)
    if to_split:
        X_train, X_test, y_train, y_test, _ = load_data(path, to_split=to_split)
    else:
        X_train, y_train, _ = load_data(path, to_split=to_split)

    with measure_time('Training...'):
        logit_model = LogisticRegression(random_state=42)
        logit_model.fit(X_train, y_train)
        #joblib.dump(logit_model, filename) 
        
    y_pred = logit_model.predict(X_train)
    print("=================================================================")
    print("Logistic Regression Training set accuracy: {}".format(accuracy_score(y_train, y_pred)))
    print("=================================================================")
    
    if to_split:
        y_pred = logit_model.predict(X_test)
        print("Logistic Regression Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("=================================================================")


if __name__ == "__main__":
    path = "_Data/Training_dataset/training_data_weight06_+surface_weighting_min20matches.csv"

    # selected_features = ['Same_handedness',
    #                      'age_diff',
    #                      'rank_diff',
    #                      'rank_points_diff',
    #                      'Win%_diff',
    #                      'bestof_diff',
    #                      'minutes_diff',
    #                      'svpt%_diff',
    #                      '1st_serve%_diff',
    #                      '1st_serve_won%_diff',
    #                      '2nd_serve_won%_diff',
    #                      'ace%_diff',
    #                      'df%_diff',
    #                      'bp_faced%_diff',
    #                      'bp_saved%_diff']
    
    # selected_features = ['age_diff',
    #                      'rank_diff',
    #                      'rank_points_diff',
    #                      'Win%_diff',
    #                      'bestof_diff',
    #                      '1st_serve_won%_diff',
    #                      '2nd_serve_won%_diff',
    #                      'bp_faced%_diff']
    
    selected_features = ['Same_handedness',
                         'age_diff',
                         'Win%_diff',
                         'bestof_diff',
                         'minutes_diff',
                         'svpt%_diff',
                         '1st_serve%_diff',
                         '1st_serve_won%_diff',
                         '2nd_serve_won%_diff',
                         'ace%_diff',
                         'df%_diff',
                         'bp_faced%_diff',
                         'bp_saved%_diff']
    
    train(path, to_split=True, selected_features=selected_features)
