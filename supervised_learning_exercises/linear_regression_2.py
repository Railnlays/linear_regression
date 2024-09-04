from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Data preparation
X = np.array([294, 314, 383, 402, 475, 786]).reshape(-1, 1) 
y = np.array([634, 728, 819, 938, 1136, 1317])

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Output the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Predicting new values
num_employees = np.array([900]).reshape(-1, 1)
predicted_sales = model.predict(num_employees)
print(f"Predicted sales for {num_employees.flatten()[0]} employees: {predicted_sales[0]}")

# Plotting the data and the regression line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Fitted line')
plt.xlabel('Number of Employees')
plt.ylabel('Annual Sales')
plt.legend()
plt.show()
