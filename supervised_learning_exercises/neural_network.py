import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Función para procesar los precios de los coches
def preprocess_price(price):
    if 'Lakh' in price:
        return float(price.replace(' Lakh', '').replace(',', '')) * 100000 * 0.011  # Convertir a euros
    elif 'Crore' in price:
        return float(price.replace(' Crore', '').replace(',', '')) * 10000000 * 0.011  # Convertir a euros
    else:
        return float(price.replace(',', '')) * 0.011  # Convertir a euros

# Cargar los datos
data = pd.read_csv('car_price.csv')

# Preprocesar los datos
data['car_prices_in_rupee'] = data['car_prices_in_rupee'].apply(preprocess_price)
data['kms_driven'] = data['kms_driven'].str.replace(' kms', '').str.replace(',', '').astype(float)
data['engine'] = data['engine'].str.replace(' cc', '').astype(float)
data['Seats'] = data['Seats'].str.replace(' Seats', '').astype(int)

# Codificar variables categóricas
label_encoder = LabelEncoder()
data['fuel_type'] = label_encoder.fit_transform(data['fuel_type'])
data['transmission'] = label_encoder.fit_transform(data['transmission'])
data['ownership'] = label_encoder.fit_transform(data['ownership'])
data['manufacture'] = label_encoder.fit_transform(data['manufacture'])

# Separar características y etiquetas
X = data[['kms_driven', 'fuel_type', 'transmission', 'ownership', 'manufacture', 'engine', 'Seats']]
y = data['car_prices_in_rupee']

# Normalizar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el conjunto de datos en train (60%), test (20%) y validation (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Crear el modelo de red neuronal
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Activación lineal por defecto
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_valid, y_valid))

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test, y_test)
print(f'Pérdida en el conjunto de prueba: {loss}')

# Predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular métricas adicionales
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error absoluto medio (MAE) en el conjunto de prueba: {mae:.2f} Euros')
print(f'Coeficiente de determinación (R²) en el conjunto de prueba: {r2:.2f}')

# Predecir el precio de un nuevo coche
new_car = np.array([[50000, 1, 0, 0, 1, 2000, 5]])  # Ejemplo: kms_driven=50000, fuel_type=1, transmission=0, ownership=0, manufacture=1, engine=2000, Seats=5
new_car = scaler.transform(new_car)
predicted_price = model.predict(new_car)
print(f'Precio predicho para el nuevo coche: {predicted_price[0][0]:.2f} Euros')