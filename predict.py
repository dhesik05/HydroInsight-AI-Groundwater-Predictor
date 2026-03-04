import pickle
import numpy as np

model = pickle.load(open("model/groundwater_model.pkl","rb"))

rainfall = float(input("Enter rainfall: "))
temperature = float(input("Enter temperature: "))
soil_moisture = float(input("Enter soil moisture: "))
humidity = float(input("Enter humidity: "))

data = np.array([[rainfall,temperature,soil_moisture,humidity]])

prediction = model.predict(data)

print("Predicted Groundwater Level:", prediction[0])