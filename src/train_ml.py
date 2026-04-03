# ==========================================
# HEART DISEASE MODEL TRAINING
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import joblib
import os

print("Loading Heart Disease Data...")
# 1. Load Data (Pointing to the data folder)
df = pd.read_csv('Data/heart.csv')

# 2. Preprocess Data
df.fillna(df.median(), inplace=True)

# One-Hot Encoding
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training Random Forest Model...")
# 3. Train Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Test the model
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {sklearn.metrics.accuracy_score(y_test, y_pred):.2f}")

# 4. Save the Model and Scaler
# Make sure the models directory exists
os.makedirs('models', exist_ok=True)

joblib.dump(rf_model, "models/heart_disease_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Heart Disease Model and Scaler saved successfully in models/ folder!")