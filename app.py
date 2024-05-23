import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the pre-trained Random Forest model
rf_model = pickle.load(open('rf_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Log the received JSON data
    print("Received JSON data:", request.json)
    
    if not request.json or 'data' not in request.json:
        return jsonify({'error': 'Invalid input, please provide data in JSON format'}), 400
    
    data = request.json['data']
    
    # Log the parsed data and its keys
    print("Parsed data:", data)
    print("Keys in parsed data:", list(data.keys()))
    
    # Define the required keys
    required_keys = ['Gender', 'AGE', 'Height_cm', 'Weight_kg', 'BMI', 'Obesity_Class']
    
    # Check for missing keys
    for key in required_keys:
        if key not in data:
            return jsonify({'error': f'Missing key: {key}'}), 400
    
    try:
        input_data = [float(data[key]) for key in required_keys]
    except ValueError:
        return jsonify({'error': 'Invalid input types, please ensure all inputs are numeric'}), 400
    
    output = rf_model.predict([input_data])[0]

    # Interpret the prediction
    if output < 20:
        vitamin_d_status = "Deficient"
    elif output < 30:
        vitamin_d_status = "Insufficient"
    elif output <= 100:
        vitamin_d_status = "Sufficient"
    else:
        vitamin_d_status = "Upper Safety Limit"

    return jsonify({'Vitamin_D_Level': output, 'Status': vitamin_d_status})

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    
    if len(data) != 6:
        return render_template("home.html", prediction_text="Invalid input data. Please provide all six inputs.")

    output = rf_model.predict([data])[0]

    # Interpret the prediction
    if output < 20:
        vitamin_d_status = "Deficient"
    elif output < 30:
        vitamin_d_status = "Insufficient"
    elif output <= 100:
        vitamin_d_status = "Sufficient"
    else:
        vitamin_d_status = "Upper Safety Limit"

    return render_template("home.html", prediction_text=f"Predicted Vitamin D Level: {output} ({vitamin_d_status})")

if __name__ == "__main__":
    app.run(debug=True)





