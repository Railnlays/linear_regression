import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Datos de entrenamiento (Nota, Gusto por la Ciencia)
X = np.array([
    [9, 8],
    [6, 7],
    [7, 6],
    [8, 9],
    [5, 5],
    [6, 8],
    [7, 9],
    [4, 4],
    [3, 3],
    [8, 7],
    [6, 9],
    [9, 7],   
    [5, 6],   
    [7, 8],   
    [8, 5],   
    [4, 9],   
    [7, 7],
    [3, 6],
    [5, 8],   
    [6, 4] 
])

# Carreras elegidas: 0 = Profesor, 1 = Ingeniería Civil, 2 = Informática, 3 = Ciencias
y = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 3, 1, 2, 0, 3, 2, 3, 2, 0, 1, 0])

# Escalamos los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenamos el modelo de regresión logística con Softmax (multiclase)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_scaled, y)

# Ejemplo de predicción con un nuevo alumno
new_student = np.array([[7, 8]])  # Nota: 7, Gusto por la Ciencia: 8
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)
prediction_proba = model.predict_proba(new_student_scaled)

# Mostrar el resultado
careers = ["Profesor", "Ingeniería Civil", "Informática", "Ciencias"]
print(f"Predicción para el alumno con Nota {new_student[0][0]} y Gusto por la Ciencia {new_student[0][1]}:")
print(f"Carrera sugerida: {careers[prediction[0]]}")
print(f"Probabilidades: {prediction_proba}")

# Visualización
plt.figure(figsize=(10, 6))

# Crear una malla para los contornos
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predecir sobre la malla para obtener las áreas de decisión
Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Dibujar los contornos de las regiones
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

# Graficar los datos de entrenamiento
for i, career in enumerate(careers):
    plt.scatter(X[y == i][:, 0], X[y == i][:, 1], label=career)

# Graficar el nuevo punto
plt.scatter(new_student[0][0], new_student[0][1], c='yellow', edgecolors='k', s=200, label='Nuevo alumno (Predicción)', marker='*')

# Etiquetas y leyenda
plt.xlabel('Nota')
plt.ylabel('Gusto por la Ciencia')
plt.legend()
plt.title('Clasificación de Carreras usando Softmax')
plt.show()

