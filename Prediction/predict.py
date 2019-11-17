from sklearn.externals import joblib
import numpy as np
import flask


app = flask.Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict(coordinates):
    """
    By taking the coordinates as inputs, predict the clustering class from:
        1. 'not_crowded' : between 0-3 people
        2. 'usual': between 3-21 people
        3. 'crowded': between 21-109 people
        4. 'very_crowded': more than 109 people
    Input: [[lat, lon]]
    """
    # Load the models from disk
    kmeans = joblib.load('models/kmeans.pkl')
    model = joblib.load('models/svc.pkl')
    
    # Predict the clustering id given coordinates
    cluster = kmeans.predict(coordinates)
    
    # Predict the crowdness given the clustering id
    crowdness = model.predict([cluster])
    
    # Return crowdness
    return crowdness[0]

app.run()
