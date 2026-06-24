# 🏠 House Rate Prediction

A Machine Learning project that predicts house prices based on features like area, bedrooms, bathrooms, location, and more — using **Linear Regression** and **Random Forest Regressor**.

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Type** | Regression (Supervised Learning) |
| **Algorithm** | Random Forest Regressor (Best), Linear Regression (Baseline) |
| **Language** | Python 3 |
| **Libraries** | scikit-learn, pandas, numpy, matplotlib, seaborn |

---

## 🏗️ Features Used

| Feature | Description |
|---------|-------------|
| `area_sqft` | Total area of the house in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_of_house` | Age of the house in years |
| `distance_to_city_km` | Distance from city center in km |
| `garage` | Garage availability (0 = No, 1 = Yes) |
| `location` | Location type (Urban / Suburban / Rural) |

---

## 📂 Project Structure

```
rate-prediction/
│
├── house_rate_prediction.py       # Main ML code
├── house_prediction_dashboard.png # Output visualization
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/rate-prediction.git
cd rate-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Project
```bash
python house_rate_prediction.py
```

---

## 📊 Expected Output

```
=====================================================
       HOUSE RATE PREDICTION PROJECT
=====================================================

📊 Dataset Overview:
   area_sqft  bedrooms  bathrooms  ...  price

MODEL EVALUATION RESULTS
📌 Linear Regression
   MAE  : ₹XX,XXX.XX
   RMSE : ₹XX,XXX.XX
   R²   : XX.XX%

📌 Random Forest Regressor
   MAE  : ₹XX,XXX.XX
   RMSE : ₹XX,XXX.XX
   R²   : XX.XX%

PREDICT A NEW HOUSE PRICE
💰 Predicted Price: ₹X,XX,XXX.XX
```

---

## 📈 Dashboard Visualization

The script generates a 4-panel dashboard:
1. **Actual vs Predicted Price** (scatter plot)
2. **Feature Importance** (bar chart)
3. **Price Distribution** (histogram)
4. **Model Comparison** (R² score comparison)

---

## 🧠 Key Concepts

- **MAE** → Average error in prediction (lower = better)
- **RMSE** → Penalizes large errors more (lower = better)
- **R² Score** → How well the model fits (closer to 100% = better)
- **Random Forest** → Multiple decision trees working together for better accuracy

---

## 👨‍💻 Author

**Nagulan**  
AIML Engineering Student  
Project: House Rate Prediction

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
