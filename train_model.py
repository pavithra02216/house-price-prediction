# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Step 1: Load dataset (make sure train.csv is in your project folder)
df = pd.read_csv("train.csv")

# Step 2: Select top 10 features
features = [
    'GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars', 'GarageArea',
    'TotalBsmtSF', 'YearBuilt', 'OverallQual', 'KitchenQual', 'LotArea'
]
X = df[features]
y = df['SalePrice']

# Step 3: Handle categorical variable (KitchenQual)
X = pd.get_dummies(X, drop_first=True)

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 6: Save model and features
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(features, open("features.pkl", "wb"))

print("✅ Model trained and saved as model.pkl with top 10 features!")
