

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 🎨 Use a modern Seaborn style
sns.set(style="whitegrid", palette="pastel")

# Load the data from the CSV file
data = pd.read_csv("home_dataset.csv")

# Extract the features and target variables
house_sizes = data["HouseSize"].values
house_prices = data["HousePrice"].values

# Visualising the data
plt.figure(figsize=(10, 6))
plt.scatter(house_sizes, house_prices, marker="o", color="#FFB6C1", edgecolor="black", s=80, alpha=0.7)
plt.title("House Prices vs. House Size", fontsize=16, fontweight="bold")
plt.xlabel("House Size (sq.ft)", fontsize=12)
plt.ylabel("House Price (£)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
sns.despine()
plt.tight_layout()
plt.show()

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)

# Reshaping the data for NumPy
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Creating and training the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predicting prices for the test set
predictions = model.predict(x_test)

# Calculate R² score
r2 = r2_score(y_test, predictions)

# Visualising the predictions data
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, marker="o", color="#FFB6C1", edgecolor="black", s=80, alpha=0.7, label="Actual Prices")
plt.plot(x_test, predictions, color="#4169E1", linewidth=2.5, label="Predicted Prices")
plt.title("Dumbo Property Price Prediction with Linear Regression", fontsize=16, fontweight="bold")
plt.xlabel("House Size (sq.ft)", fontsize=12)
plt.ylabel("House Price (millions (£))", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)

# Annotate R² score on plot
plt.text(0.05, 0.95,
         f"R² = {r2:.3f}",
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

sns.despine()
plt.tight_layout()
plt.show()


