import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction App")

# Input fields
sqft = st.number_input("Enter square feet:", min_value=300, max_value=10000, step=50)
bedrooms = st.number_input("Number of Bedrooms:", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms:", min_value=1, max_value=10, step=1)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[sqft, bedrooms, bathrooms]],
                              columns=["sqft", "bedrooms", "bathrooms"])
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ₹{prediction[0]:,.2f}")
