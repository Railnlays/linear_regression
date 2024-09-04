import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Crear un pequeño conjunto de datos de ejemplo
# Características: [Edad, Tamaño del Tumor]
# Etiquetas: 1 = Maligno, 0 = Benigno
X = np.array([
    [25, 3],
    [30, 1],
    [45, 6],
    [50, 8],
    [35, 2],
    [60, 7],
    [70, 9],
    [65, 5],
    [40, 4],
    [55, 7]
])

y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 1])  # Etiquetas correspondientes

# Inicializar y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X, y)

# Ejemplo de predicción con nuevas características
new_example = np.array([[58, 6]])  # Edad: 58, Tamaño del Tumor: 6 cm
prediction = model.predict(new_example)

# Mostrar el resultado
print(f"Predicción para el ejemplo con Edad {new_example[0][0]} y Tamaño de Tumor {new_example[0][1]}:")
print("Maligno" if prediction[0] == 1 else "Benigno")

# Visualización
plt.figure(figsize=(10, 6))

# Graficar los datos de entrenamiento
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=100, label='Datos de entrenamiento')

# Graficar el nuevo punto
plt.scatter(new_example[0][0], new_example[0][1], c='yellow', edgecolors='k', s=200, label='Nuevo punto (Predicción)', marker='*')

# Crear una malla para la línea de decisión
xx, yy = np.meshgrid(np.arange(20, 80, 0.1), np.arange(0, 10, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar la línea de decisión
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black', label='Línea de decisión')

# Etiquetas y leyenda
plt.xlabel('Edad')
plt.ylabel('Tamaño del Tumor (cm)')
plt.legend()
plt.title('Regresión Logística: Clasificación de Tumores')
plt.show()
