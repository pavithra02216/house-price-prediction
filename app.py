import streamlit as st
import pickle
import pandas as pd

# Load model and features
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("🏠 House Price Prediction App")

# Collect inputs from user
GrLivArea = st.number_input("Above ground living area (sq ft)", min_value=100, step=50)
BedroomAbvGr = st.number_input("Number of Bedrooms", min_value=0, step=1)
FullBath = st.number_input("Number of Full Bathrooms", min_value=0, step=1)
GarageCars = st.number_input("Garage capacity (cars)", min_value=0, step=1)
GarageArea = st.number_input("Garage area (sq ft)", min_value=0, step=50)
TotalBsmtSF = st.number_input("Total basement area (sq ft)", min_value=0, step=50)
YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, step=1)
OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
KitchenQual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"])  # categorical
LotArea = st.number_input("Lot Area (sq ft)", min_value=500, step=100)

# Put all inputs in dataframe
input_data = pd.DataFrame([[
    GrLivArea, BedroomAbvGr, FullBath, GarageCars, GarageArea,
    TotalBsmtSF, YearBuilt, OverallQual, KitchenQual, LotArea
]], columns=features)

# Convert categorical to dummies
input_data = pd.get_dummies(input_data)
# Align with training features
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model.feature_names_in_]

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    # st.success(f"🏡 Predicted House Price: ${prediction:,.2f}")
    # Example: prediction display in app.py

    if prediction < 50000:
        st.success(f"💸 Affordable House! Estimated Price: ₹{prediction:,.2f}")
    elif 50000 <= prediction < 150000:
        st.warning(f"🏡 Mid-range House! Estimated Price: ₹{prediction:,.2f}")
    else:
        st.error(f"💎 Luxury House! Estimated Price: ₹{prediction:,.2f}")
    st.balloons()

