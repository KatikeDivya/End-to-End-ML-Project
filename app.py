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
    data = request.json['data']
    input_data = [float(x) for x in data.values()]
    output = rf_model.predict([input_data])[0]
    return jsonify({'Vitamin_D_Level': output})

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
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
