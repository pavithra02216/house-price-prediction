# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.title("🏡 House Price Prediction App")

st.write("Enter the details below to predict the house price:")

# Inputs for features
input_data = {}
for feature in features:
    if feature == "KitchenQual":  
        input_data[feature] = st.selectbox("Kitchen Quality", ["TA", "Gd", "Ex", "Fa"])
    else:
        input_data[feature] = st.number_input(feature, min_value=0, value=0)

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Handle categorical (KitchenQual)
input_df = pd.get_dummies(input_df, drop_first=True)

# Align with model training columns
missing_cols = set(model.feature_names_in_) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[model.feature_names_in_]

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")
