import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load model
model = load_model("model.h5", compile=False)

# Load encoders & scaler
with open("pkl_files/onehot_encoder_geography.pkl", "rb") as file:
    onehot_encoder_geography = pickle.load(file)

with open("pkl_files/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("Customer Churn Prediction")

# User inputs
geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Encode gender
gender_val = 1 if gender == "Male" else 0

# Create input dataframe
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_val],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode geography
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geography.get_feature_names_out(["Geography"])
)

# Combine data
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Scale data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Output
if prediction_proba > 0.5:
    st.error("🚨 The customer is likely to churn")
else:
    st.success("✅ The customer is not likely to churn")
