# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("car_data.csv")

# Show basic info
print("First 5 rows:\n", data.head())
print("\nDataset shape:", data.shape)
print("\nColumns:\n", data.columns)

# Drop 'Car_Name' as it's not useful for regression
data = data.drop('Car_Name', axis=1)

# Convert categorical data using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("R² Score:", round(r2, 3))
print("RMSE:", round(rmse, 3))

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.grid(True)
plt.tight_layout()
plt.show()

