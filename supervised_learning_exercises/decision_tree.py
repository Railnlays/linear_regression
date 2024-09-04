import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Cargar los datos
data = pd.read_csv('symptoms.csv')  # Asegúrate de guardar tus datos en este archivo

# Reemplazar valores NaN con 'none' para los síntomas que no están presentes
data = data.fillna('none')

# Codificar las variables categóricas usando One-Hot Encoding
X = pd.get_dummies(data.drop('Disease', axis=1))
y = data['Disease']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

def get_user_symptoms(symptoms):
    # Inicializar el diccionario de síntomas del usuario
    user_symptoms = {symptom: 0 for symptom in symptoms}
    
    # Convertir el índice de síntomas a lista para procesarlo
    symptoms_to_check = list(symptoms)
    
    # Procesar síntomas hasta que la lista esté vacía
    while symptoms_to_check:
        symptom = symptoms_to_check.pop(0)  # Obtener el primer síntoma de la lista
        
        if user_symptoms[symptom] == 0:  # Solo pregunta si el síntoma no ha sido marcado
            response = input(f"¿Tienes el síntoma '{symptom}'? (sí/no): ").strip().lower()
            if response in ['sí', 'si', 'yes']:
                user_symptoms[symptom] = 1
                print(f"Síntoma '{symptom}' confirmado.")
                # Eliminar el síntoma de la lista de síntomas a verificar
                symptoms_to_check = [s for s in symptoms_to_check if s != symptom]
            elif response in ['no', 'no']:
                user_symptoms[symptom] = 0
                print(f"Síntoma '{symptom}' no confirmado.")
            else:
                print("Respuesta no válida. Asumiendo 'no'.")
                user_symptoms[symptom] = 0
    
    return user_symptoms

# Obtener los nombres de los síntomas como lista
symptoms = list(X.columns)

# Obtener la entrada del usuario
print("Por favor, responde a las siguientes preguntas sobre los síntomas:")
user_symptoms = get_user_symptoms(symptoms)

# Crear un DataFrame con los síntomas del usuario
user_symptoms_df = pd.DataFrame([user_symptoms], columns=symptoms)

# Realizar la predicción
predicted_disease = model.predict(user_symptoms_df)
print(f'La enfermedad predicha para los síntomas ingresados es: {predicted_disease[0]}')