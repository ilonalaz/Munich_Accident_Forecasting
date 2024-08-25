import pickle
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the model
with open('sarima_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json()
    year = data['year']
    month = data['month']

    # Create a datetime index for the prediction date
    predict_date = pd.to_datetime(f'{year}-{month}-01')

    # Get the forecast for the given date
    forecast = model.get_forecast(steps=1)
    prediction = forecast.predicted_mean.iloc[0]

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)