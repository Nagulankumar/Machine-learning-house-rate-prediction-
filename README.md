# House Price Prediction using Linear Regression

## Project Overview

This project predicts house prices using a **Machine Learning Linear Regression model**.
The dataset contains information about different house features such as area, bedrooms, bathrooms, and other facilities. The model learns from the dataset and predicts the price of houses based on these features.

This project demonstrates the basic steps in a **machine learning workflow**, including data preprocessing, feature scaling, model training, evaluation, and visualization.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Dataset

The dataset (`dataset.csv`) contains information about houses with features such as:

* Area
* Bedrooms
* Bathrooms
* Stories
* Parking
* Main road access
* Guest room availability
* Basement
* Hot water heating
* Air conditioning
* Preferred area
* Furnishing status
* Price (Target Variable)

Categorical values such as **yes/no** and **furnishing status** are converted into numerical values so that the machine learning model can process them.

---

## Project Workflow

### 1. Import Libraries

Required libraries for data processing, visualization, and machine learning are imported.

### 2. Load Dataset

The dataset is loaded using **Pandas**.

### 3. Data Preprocessing

Categorical data is converted into numerical values:

* yes → 1
* no → 0
* furnished → 1
* semi-furnished → 2
* unfurnished → 3

### 4. Feature and Target Selection

* Features (X): All columns except price
* Target (y): Price

### 5. Feature Scaling

The **StandardScaler** is used to normalize feature values so that all features are on a similar scale.

### 6. Train-Test Split

The dataset is divided into:

* 80% training data
* 20% testing data

### 7. Model Training

A **Linear Regression model** is trained using the training dataset.

### 8. Prediction

The trained model predicts house prices for the test dataset.

### 9. Model Evaluation

Two evaluation metrics are used:

* **R² Score** – Measures how well the model fits the data.
* **Mean Squared Error (MSE)** – Measures the average prediction error.

### 10. Data Visualization

A **correlation heatmap** is created using Seaborn to visualize relationships between features.

---

## Output

The program prints:

* R² Score
* Mean Squared Error

It also displays a **feature correlation heatmap**.

---

## How to Run the Project

1. Clone the repository from GitHub.

2. Install required libraries:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Run the Python file:

```
python model.py
```

---

## Example Visualization

The project generates a **correlation heatmap** to show relationships between house features and price.

---

## Future Improvements

* Add more advanced models such as Random Forest
* Improve feature engineering
* Add model comparison
* Deploy the model as a web application

---

## Author

**Nagulan**
Artificial Intelligence and Machine Learning Student
