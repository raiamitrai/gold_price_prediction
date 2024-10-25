import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from PIL import Image

# Load the dataset
# Make sure to update the path to your CSV file
gold = pd.read_csv('gold_price_data.csv')

# Prepare the data
x = gold.drop(['Date', 'GLD'], axis=1)  # Adjust according to your dataset
y = gold['GLD']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train the model
regressor = RandomForestRegressor(n_estimators=200)
regressor.fit(x_train, y_train)

# User input for prediction
st.title('Gold Price Prediction')
st.subheader('Enter the features for prediction')

# Create input fields for each feature
user_input = {}
for column in x.columns:
    min_value = float(x[column].min())
    max_value = float(x[column].max())
    user_input[column] = st.number_input(
        f'Enter value for {column}',
        min_value=min_value,
        max_value=max_value,
        value=min_value  # Default to min_value
    )

# Convert user input to DataFrame
input_data = pd.DataFrame(user_input, index=[0])

# Prediction button
if st.button("Predict"):
    prediction = regressor.predict(input_data)
    st.write(f'Predicted GLD Price: {prediction[0]:.2f}')

# Optionally display model performance metrics
test_data_prediction = regressor.predict(x_test)
score = r2_score(y_test, test_data_prediction)
st.subheader('Model Performance')
st.write(f'R-squared Score: {score:.2f}')


# web app
st.title('Gold Price Model')
img = Image.open('goldimg.jpeg')
st.image(img,width=200,use_column_width=True)

st.subheader('Using randomforestregressor')
st.write(gold)
st.subheader('Model Performance')
st.write(score)
