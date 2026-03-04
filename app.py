from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/groundwater_model.pkl","rb"))

@app.route('/')
def home():
    return '''
    <h2>HydroInsight - Groundwater Predictor</h2>
    <form method="POST" action="/predict">
    Rainfall: <input name="rainfall"><br>
    Temperature: <input name="temperature"><br>
    Soil Moisture: <input name="soil"><br>
    Humidity: <input name="humidity"><br>
    <button type="submit">Predict</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    rainfall = float(request.form['rainfall'])
    temperature = float(request.form['temperature'])
    soil = float(request.form['soil'])
    humidity = float(request.form['humidity'])

    data = np.array([[rainfall,temperature,soil,humidity]])

    prediction = model.predict(data)

    return f"Predicted Groundwater Level: {prediction[0]}"

if __name__ == "__main__":
    app.run(debug=True)