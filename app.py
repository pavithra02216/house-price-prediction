import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('house_price_model.pkl')

st.title("🏠 House Price Prediction App")

# User inputs
size = st.number_input("Size (sq ft)", min_value=300, max_value=10000, value=1000)
rooms = st.number_input("Number of Rooms", min_value=1, max_value=20, value=3)
floor = st.number_input("Floor", min_value=0, max_value=50, value=1)
age = st.number_input("Age of House (years)", min_value=0, max_value=100, value=5)
location = st.text_input("Location", "Urban")
house_type = st.selectbox("Type of House", ["Apartment", "Villa", "Duplex"])

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'GrLivArea': [size],
        'TotRmsAbvGrd': [rooms],
        'OverallQual': [7],  # placeholder
        'YearBuilt': [2020 - age],
        'Neighborhood': [location],
        'HouseStyle': [house_type]
    })
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ${prediction:,.2f}")
