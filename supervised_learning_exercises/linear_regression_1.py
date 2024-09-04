import numpy as np
import math as math
import copy as copy
import matplotlib.pyplot as plt

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    """
    m = x.shape[0]
    predictions = w * x + b
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    return cost

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression.
    """
    m = x.shape[0]
    predictions = w * x + b
    dj_dw = np.sum((predictions - y) * x) / m
    dj_db = np.sum(predictions - y) / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b.
    """
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 100 == 0:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history, w_history

def predict(x, w, b):
    """
    Predicts the output using the linear model with parameters w and b.
    """
    return w * x + b

def desnormalizar_x(x_norm, x_min, x_max):
    return x_norm * (x_max - x_min) + x_min

def desnormalizar_y(y_norm, y_min, y_max):
    return y_norm * (y_max - y_min) + y_min

# Datos proporcionados
x = np.array([294, 314, 383, 402, 475, 786])  # Número de empleados
y = np.array([634, 728, 819, 938, 1136, 1317])  # Ventas anuales

# Normalización de los datos
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)
x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)

# Valores iniciales para w y b
w_in = 0.0
b_in = 0.0

# Parámetros del descenso por gradiente
alpha = 0.01
num_iters = 10000

# Entrenar el modelo
w_final, b_final, J_history, w_history = gradient_descent(x_norm, y_norm, w_in, b_in, compute_cost, compute_gradient, alpha, num_iters)

print(f"Final w: {w_final}, Final b: {b_final}")

# Predicción con normalización
num_employees_norm = (900 - x_min) / (x_max - x_min)
prediction_norm = predict(num_employees_norm, w_final, b_final)
prediction = desnormalizar_y(prediction_norm, y_min, y_max)

print(f"Predicted sales for {900} employees: {prediction}")

# Gráfica del costo
plt.figure(figsize=(12, 6))

# Gráfico del costo durante el entrenamiento
plt.subplot(1, 2, 1)
plt.plot(range(len(J_history)), J_history, 'b-')
plt.title("Costo durante el entrenamiento")
plt.xlabel("Número de Iteraciones")
plt.ylabel("Costo")

# Gráfico de la línea de regresión, los datos y la predicción
plt.subplot(1, 2, 2)
plt.scatter(x, y, color='red', marker='x', label='Datos reales')

# Línea de regresión
x_plot = np.linspace(x_min, x_max, 100)
x_plot_norm = (x_plot - x_min) / (x_max - x_min)
y_plot = predict(x_plot_norm, w_final, b_final)
y_plot = desnormalizar_y(y_plot, y_min, y_max)
plt.plot(x_plot, y_plot, color='blue', label='Línea de regresión')

# Añadir punto de predicción
plt.scatter(900, prediction, color='green', marker='o', s=100, label='Predicción para 500 empleados')

plt.title("Datos reales y línea de regresión")
plt.xlabel("Número de empleados")
plt.ylabel("Ventas anuales")
plt.legend()

plt.tight_layout()
plt.show()


