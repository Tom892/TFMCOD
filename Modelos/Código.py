# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:09:03 2024

@author: tommo
"""
#Random Forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os

directorio_script = os.path.dirname(__file__)

file_path = os.path.join(directorio_script, '..', 'Datos', 'Datos final tesis python.xlsx')


df = pd.read_excel(file_path)

# Convertir las columnas categóricas en numéricas

label_encoder_barrio = LabelEncoder()
df['Barrio'] = label_encoder_barrio.fit_transform(df['Barrio'])

# 'Antigüedad' con un mapeo personalizado
df['Antigüedad'] = df['Antigüedad'].map({'1 a 8 años': 1, '9 a 15 años': 2})

# Separar las variables independientes (X) y la variable dependiente (y)
X = df.drop('Precio Total', axis=1)  
y = df['Precio Total']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Random Forest
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Calcular y mostrar las métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("Random Forest - Mean Squared Error:", mse)
print("Random Forest - R^2 Score:", r2)
print("Random Forest - Mean Absolute Error (MAE):", mae)
print("Random Forest - Root Mean Squared Error (RMSE):", rmse)
print("Random Forest - Mean Absolute Percentage Error (MAPE):", mape, "%")

# Esto para la interfaz gráfica
import pickle

ruta_archivo_modelo_rf = os.path.join(directorio_script, 'modelo_rf_entrenado.pkl')

# Guardar el modelo en un archivo .pkl
with open(ruta_archivo_modelo_rf, 'wb') as archivo:
    pickle.dump(rf_model, archivo)
    
    
    
    
#Regresión Lineal Múltiple

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


df = pd.read_excel(file_path)

# Convertir las columnas categóricas en numéricas
label_encoder = LabelEncoder()
df['Barrio'] = label_encoder.fit_transform(df['Barrio'])
df['Antigüedad'] = df['Antigüedad'].map({'1 a 8 años': 1, '9 a 15 años': 2})

# Separar las variables independientes (X) y la variable dependiente (y)
X = df.drop('Precio Total', axis=1)  
y = df['Precio Total']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal múltiple
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred_lr = lr_model.predict(X_test)

# Calcular y mostrar las métricas 
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100

print("Regresión - Mean Squared Error (MSE):", mse_lr)
print("Regresión - R^2 Score:", r2_lr)
print("Regresión - Mean Absolute Error (MAE):", mae_lr)
print("Regresión - Root Mean Squared Error (RMSE):", rmse_lr)
print("Regresión - Mean Absolute Percentage Error (MAPE):", mape_lr, "%")

# Esto para la interfaz gráfica
import pickle

ruta_archivo_modelo_lr = os.path.join(directorio_script, 'modelo_lr_entrenado.pkl')

with open(ruta_archivo_modelo_lr, 'wb') as archivo:
    pickle.dump(lr_model, archivo)









# Redes neuronales artificiales

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np


df = pd.read_excel(file_path)

# Convertir las columnas categóricas en numéricas
label_encoder_barrio = LabelEncoder()
df['Barrio'] = label_encoder_barrio.fit_transform(df['Barrio'])

df['Antigüedad'] = df['Antigüedad'].map({'1 a 8 años': 1, '9 a 15 años': 2})

# Separar las variables independientes (X) y la variable dependiente (y)
X = df.drop('Precio Total', axis=1)  
y = df['Precio Total']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de redes neuronales
nn_model = MLPRegressor(hidden_layer_sizes=(100,100),
                        activation='relu',
                        solver='adam',
                        max_iter= 5000,
                        learning_rate_init=0.1,
                        alpha=0.005,  
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        random_state=42)
nn_model.fit(X_train_scaled, y_train)
# Hacer predicciones en el conjunto de prueba
y_pred_nn = nn_model.predict(X_test_scaled)

# Calcular y mostrar las métricas 
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
mape_nn = np.mean(np.abs((y_test - y_pred_nn) / y_test)) * 100

print("ANN - Mean Squared Error (MSE):", mse_nn)
print("ANN - R^2 Score:", r2_nn)
print("ANN - Mean Absolute Error (MAE):", mae_nn)
print("ANN - Root Mean Squared Error (RMSE):", rmse_nn)
print("ANN - Mean Absolute Percentage Error (MAPE):", mape_nn, "%")

# Esto para la interfaz gráfica
import pickle

ruta_archivo_modelo_nn = os.path.join(directorio_script, 'modelo_nn_entrenado.pkl')


with open(ruta_archivo_modelo_nn, 'wb') as archivo:
    pickle.dump(nn_model, archivo)
    
    
    
    
    
    
    
    
    #Interfaz
    
    import tkinter as tk
    from tkinter import ttk, messagebox
    import joblib
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd


    df = pd.read_excel(file_path)

    # Convertir las columnas categóricas en numéricas
    label_encoder_barrio = LabelEncoder()
    label_encoder_barrio.fit(df['Barrio'])

    mapeo_antiguedad = {"1 a 8 años": 1, "9 a 15 años": 2}

    # Crear la ventana
    root = tk.Tk()
    root.title("Predicción de precios de inmuebles en Chapinero")

    # Crear etiquetas y campos de entrada para cada variable
    labels = ['Barrio', 'Habitaciones', 'Baños', 'Parqueaderos', 'Área construida', 'Antigüedad', 'Piso', 'Admón', 'Distancia Country', 'Distancia Andino', 'Distancia a parque 93', 'Distancia parque virrey']
    entries = {}

    barrios = df['Barrio'].unique().tolist()

    for i, label in enumerate(labels):
        tk.Label(root, text=label).grid(row=i, column=0)

        if label == 'Barrio':
            combobox = ttk.Combobox(root, values=barrios)
            combobox.grid(row=i, column=1)
            entries[label] = combobox
        elif label == 'Antigüedad':
            antiguedades = ["1 a 8 años", "9 a 15 años"]
            combobox = ttk.Combobox(root, values=antiguedades)
            combobox.grid(row=i, column=1)
            entries[label] = combobox
        else:
            entry = tk.Entry(root)
            entry.grid(row=i, column=1)
            entries[label] = entry

    # Crear un menú desplegable para la selección del modelo
    model_selection = ttk.Combobox(root, values=["Random Forest", "Regresión Lineal Múltiple", "Redes Neuronales"])
    model_selection.grid(row=len(labels), column=1)
    model_selection.set("Seleccione un modelo")

 

    def predecir():
        try:
            datos = []
            for label in labels:
                valor = entries[label].get()
                if label == 'Barrio':
                    valor = label_encoder_barrio.transform([valor])[0]
                elif label == 'Antigüedad':
                    valor = mapeo_antiguedad[valor]
                else:
                    valor = float(valor)
                datos.append(valor)
    
            modelo_seleccionado = model_selection.get()
            if modelo_seleccionado == "Random Forest":
                modelo = rf_model
            elif modelo_seleccionado == "Regresión Lineal Múltiple":
                modelo = lr_model
            elif modelo_seleccionado == "Redes Neuronales":
                modelo = nn_model
            else:
                messagebox.showwarning("Advertencia", "Seleccione un modelo válido")
                return
    
            prediccion = modelo.predict([datos])
            resultado_formateado = "{:,.2f}".format(prediccion[0]) 
            resultado.set(f"Predicción de Precio: ${resultado_formateado}")
        except ValueError as e:
            messagebox.showerror("Error", f"Error en la entrada de datos: {e}")



    # Crear un botón para realizar la predicción
    tk.Button(root, text="Predecir", command=predecir).grid(row=len(labels)+1, column=1)

    # Se crea un lugar para mostrar el resultado
    resultado = tk.StringVar()
    resultado_label = tk.Label(root, textvariable=resultado, font=("Helvetica", 14), fg="blue")
    resultado_label.grid(row=len(labels)+2, column=1, sticky="w")

    root.mainloop()
    
    