
import time, datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier


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
    y = df['crowd_class'].values.squeeze()

    # Get the input values
    feature = 'cluster'
    X = df[feature].values.squeeze()
    X = X.reshape(-1, 1)
    
    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if to_split:
        return X_train, X_test, y_train, y_test
    else:
        return X, y
    
    
def train(path, to_split=True):
    """
    Train the model.
    """
    filename = "../models/knn.pkl"
    
    # Load the training (and testing) set(s)
    if to_split:
        X_train, X_test, y_train, y_test = load_data(path, to_split=to_split)
    else:
        X_train, y_train = load_data(path, to_split=to_split)

    with measure_time('Training...'):
        model = KNeighborsClassifier(n_neighbors=100)
        model.fit(X_train, y_train)
        joblib.dump(model, filename) 
        
    y_pred = model.predict(X_train)
    print("=================================================================")
    print("Knn Training set accuracy: {}".format(accuracy_score(y_train, y_pred)))
    print("=================================================================")
    
    if to_split:
        y_pred = model.predict(X_test)
        print("Knn Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("=================================================================")


# Train our model
path = './data/cleaned_HSL_data.csv'
train(path)
