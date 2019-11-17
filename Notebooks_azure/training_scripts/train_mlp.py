
import time, datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV


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
    
    
def tune_hyperparameter(path):
    """
    Get the best hyperparameters.
    """
   # Load the training set
    X, y = load_data(path, to_split=False)
        
    # Create the random grid
    random_grid = {'hidden_layer_sizes': [(20,), (50,), (100,), (150,)],
                    'activation': ['tanh', 'relu', 'logistic', 'identity'],
                    'learning_rate_init': [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
                    'learning_rate': ['constant','adaptive'],
                    'momentum': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    mlp = MLPClassifier(solver='sgd', early_stopping=True)
    # Random search of parameters, using 5 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    mlp_random = RandomizedSearchCV(estimator = mlp,
                                   param_distributions = random_grid,
                                   n_iter = 100,
                                   cv = 5,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs = -1)
    # Fit the random search model
    mlp_random.fit(X, y)

    # Return optimal parameters
    return mlp_random.best_params_
    
    
def create_estimator(path, params, to_split=True):
    """
    Train the model.
    """
    filename = "models/mlp.pkl"
    
    # Load the training (and testing) set(s)
    if to_split:
        X_train, X_test, y_train, y_test = load_data(path, to_split=to_split)
    else:
        X_train, y_train = load_data(path, to_split=to_split)
        
    # Extract parameters
    hidden_layer_sizes = params['hidden_layer_sizes']
    learning_rate_init = params['learning_rate_init']
    learning_rate = params['learning_rate']
    activation = params['activation']
    momentum = params['momentum']

    with measure_time('Training...'):
        model = MLPClassifier(solver='sgd', 
                                hidden_layer_sizes=hidden_layer_sizes, 
                                early_stopping=True,
                                learning_rate_init=learning_rate_init,
                                learning_rate = learning_rate,
                                activation= activation,
                                momentum= momentum)
        model.fit(X_train, y_train)
        joblib.dump(model, filename) 
        
    y_pred = model.predict(X_train)
    print("=================================================================")
    print("MLP Training set accuracy: {}".format(accuracy_score(y_train, y_pred)))
    print("=================================================================")
    
    if to_split:
        y_pred = model.predict(X_test)
        print("MLP Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("=================================================================")



# Hypertune model
path = './data/cleaned_HSL_data.csv'
params = tune_hyperparameter(path)
print("Best parameters", params)

# Train model on best parameters
create_estimator(path, params)
