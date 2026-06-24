# ============================================================
# House Rate Prediction using Random Forest Regressor
# Author: Nagulan
# Description: Predicts house prices based on features like
#              area, bedrooms, bathrooms, location, etc.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Create Sample Dataset
# (In real project, use: df = pd.read_csv("house_data.csv"))
# ============================================================

np.random.seed(42)
n_samples = 500

data = {
    'area_sqft':      np.random.randint(500, 5000, n_samples),
    'bedrooms':       np.random.randint(1, 6, n_samples),
    'bathrooms':      np.random.randint(1, 4, n_samples),
    'age_of_house':   np.random.randint(0, 50, n_samples),
    'distance_to_city_km': np.random.uniform(1, 30, n_samples),
    'garage':         np.random.randint(0, 2, n_samples),   # 0 = No, 1 = Yes
    'location':       np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
}

# Generate realistic house prices based on features
price = (
    data['area_sqft'] * 150 +
    data['bedrooms']  * 20000 +
    data['bathrooms'] * 15000 -
    data['age_of_house'] * 1000 -
    data['distance_to_city_km'] * 3000 +
    data['garage'] * 25000 +
    np.where(np.array(data['location']) == 'Urban', 80000,
    np.where(np.array(data['location']) == 'Suburban', 40000, 0)) +
    np.random.randint(-20000, 20000, n_samples)  # random noise
)

data['price'] = price
df = pd.DataFrame(data)

print("=" * 55)
print("       HOUSE RATE PREDICTION PROJECT")
print("=" * 55)

# ============================================================
# STEP 2: Exploratory Data Analysis (EDA)
# ============================================================

print("\n📊 Dataset Overview:")
print(df.head())
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Statistics:\n", df.describe().round(2))

# ============================================================
# STEP 3: Data Preprocessing
# ============================================================

# Encode 'location' column (text → number)
label_encoder = LabelEncoder()
df['location_encoded'] = label_encoder.fit_transform(df['location'])

# Features (input) and Target (output)
feature_columns = ['area_sqft', 'bedrooms', 'bathrooms', 'age_of_house',
                   'distance_to_city_km', 'garage', 'location_encoded']

X = df[feature_columns]
y = df['price']

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Training samples : {X_train.shape[0]}")
print(f"✅ Testing samples  : {X_test.shape[0]}")

# ============================================================
# STEP 4: Train Models
# ============================================================

# --- Model 1: Linear Regression (baseline) ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# --- Model 2: Random Forest Regressor (recommended) ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ============================================================
# STEP 5: Evaluate Models
# ============================================================

def evaluate_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n📌 {name}")
    print(f"   MAE  (Mean Absolute Error) : ₹{mae:,.2f}")
    print(f"   RMSE (Root Mean Sq. Error) : ₹{rmse:,.2f}")
    print(f"   R²   (Accuracy Score)      : {r2:.4f} ({r2*100:.2f}%)")
    return mae, rmse, r2

print("\n" + "=" * 55)
print("          MODEL EVALUATION RESULTS")
print("=" * 55)
lr_mae, lr_rmse, lr_r2 = evaluate_model("Linear Regression", y_test, lr_pred)
rf_mae, rf_rmse, rf_r2 = evaluate_model("Random Forest Regressor", y_test, rf_pred)

print("\n🏆 Best Model: Random Forest Regressor" if rf_r2 > lr_r2 else "\n🏆 Best Model: Linear Regression")

# ============================================================
# STEP 6: Predict a New House Price
# ============================================================

print("\n" + "=" * 55)
print("          PREDICT A NEW HOUSE PRICE")
print("=" * 55)

new_house = pd.DataFrame([{
    'area_sqft':           1800,
    'bedrooms':            3,
    'bathrooms':           2,
    'age_of_house':        5,
    'distance_to_city_km': 10,
    'garage':              1,
    'location_encoded':    label_encoder.transform(['Urban'])[0]
}])

predicted_price = rf_model.predict(new_house)[0]
print(f"\n🏠 House Details:")
print(f"   Area        : 1800 sq.ft")
print(f"   Bedrooms    : 3 | Bathrooms: 2")
print(f"   Age         : 5 years")
print(f"   Distance    : 10 km from city")
print(f"   Garage      : Yes")
print(f"   Location    : Urban")
print(f"\n💰 Predicted Price: ₹{predicted_price:,.2f}")

# ============================================================
# STEP 7: Visualizations
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('House Rate Prediction - Analysis Dashboard', fontsize=15, fontweight='bold')

# Plot 1: Actual vs Predicted (Random Forest)
axes[0, 0].scatter(y_test, rf_pred, alpha=0.5, color='steelblue', edgecolors='navy', s=40)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
axes[0, 0].set_xlabel('Actual Price (₹)')
axes[0, 0].set_ylabel('Predicted Price (₹)')
axes[0, 0].set_title('Actual vs Predicted Price (Random Forest)')
axes[0, 0].legend()

# Plot 2: Feature Importance
importances = rf_model.feature_importances_
feature_names = feature_columns
sorted_idx = np.argsort(importances)[::-1]
axes[0, 1].barh([feature_names[i] for i in sorted_idx],
                [importances[i] for i in sorted_idx],
                color='coral', edgecolor='darkred')
axes[0, 1].set_title('Feature Importance (Random Forest)')
axes[0, 1].set_xlabel('Importance Score')

# Plot 3: Price Distribution
axes[1, 0].hist(df['price'], bins=30, color='mediumseagreen', edgecolor='darkgreen', alpha=0.8)
axes[1, 0].set_xlabel('House Price (₹)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('House Price Distribution')

# Plot 4: Model Comparison Bar Chart
model_names  = ['Linear Regression', 'Random Forest']
r2_scores    = [lr_r2 * 100, rf_r2 * 100]
bar_colors   = ['#FF6B6B', '#4ECDC4']
bars = axes[1, 1].bar(model_names, r2_scores, color=bar_colors, edgecolor='black', width=0.4)
axes[1, 1].set_ylabel('R² Score (%)')
axes[1, 1].set_title('Model Comparison (R² Score)')
axes[1, 1].set_ylim(0, 110)
for bar, score in zip(bars, r2_scores):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{score:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('house_prediction_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Dashboard saved as 'house_prediction_dashboard.png'")
print("\n🎉 Project Complete!")
