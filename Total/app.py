from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/howitworks')
def howitworks():
    return render_template('howitworks.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting form data
        area_worst = float(request.form['area_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        concave_points_mean = float(request.form['concave_points_mean'])
        radius_worst = float(request.form['radius_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])

        # Preparing the features for prediction
        features = np.array([[area_worst, concave_points_worst, concave_points_mean, radius_worst, perimeter_worst]])
        scaled_features = scaler.transform(features)

        # Making a prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)[0]

        # Prepare the response data
        response_data = {
            'prediction': 'Cancer' if prediction[0] == 1 else 'No Cancer',
            'benign_percentage': round(prediction_proba[0] * 100, 2),
            'malignant_percentage': round(prediction_proba[1] * 100, 2)
        }

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
