import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Especifica la ruta de tu archivo de texto
ruta_archivo_txt = 'ALL_MULT.txt'

# Usa pd.read_csv con el delimitador adecuado (coma en este caso)
df = pd.read_csv(ruta_archivo_txt, delimiter=',')

# Asigna los nombres de las columnas
df.columns = ["ID", "Temperature", "Humidity", "UV", "Voltage", "Current", "Illuminance", "ClientIP", "SensorID", "DateTime"]


# Supongamos que tienes una nueva columna llamada 'Voltaje_Categoria' que representa la categoría del voltaje
# Si no tienes esta columna, puedes crearla basada en algún criterio
df['Voltaje_Categoria'] = pd.cut(df['Voltage'], bins=[0, 15, 25, 100], labels=['Bajo', 'Medio', 'Alto'])

# Elegir las características que se utilizarán para la predicción
features = ["Temperature", "Humidity", "UV", "Current", "Illuminance"]

# Eliminar filas con valores NaN
df = df.dropna()

# Seleccionar características y variable de destino
X = df[features].values
y = df['Voltaje_Categoria'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el clasificador Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar el modelo
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print("Matriz de Confusión:")
print(cm)
print("\nPrecisión del Modelo:", ac)
