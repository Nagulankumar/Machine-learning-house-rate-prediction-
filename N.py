import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('dataset.csv')

cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
df[cols] = df[cols].replace({'yes':1, 'no':0})


df['furnishingstatus'] = df['furnishingstatus'].replace({
    'furnished':1,
    'semi-furnished':2,
    'unfurnished':3
})

X = df.drop('price', axis=1)
y = df['price']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 Score:", r2)
print("Mean Squared Error:", mse)


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()