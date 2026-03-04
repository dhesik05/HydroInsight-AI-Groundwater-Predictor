import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("../dataset/groundwater_data.csv")

X = data[['rainfall','temperature','soil_moisture','humidity']]
y = data['groundwater_level']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train,y_train)

# Save model
pickle.dump(model, open("groundwater_model.pkl","wb"))

print("Model trained and saved successfully.")