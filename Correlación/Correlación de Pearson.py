# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 11:32:18 2024

@author: tommo
"""
# Importación de bibliotecas necesarias para obtener la correlación de Pearson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# Obtener la ruta al directorio actual del script
directorio_script = os.path.dirname(__file__)


# Construir la ruta al archivo de datos, relativa al directorio del script
file_path = os.path.join(directorio_script, '..', 'Datos', 'Datos final tesis python.xlsx')


# Cargar los datos desde un archivo Excel
df = pd.read_excel(file_path)


# Transformar la columna 'Antigüedad' en valores numéricos para facilitar el análisis
# Mapeo: '1 a 8 años' a 1 y '9 a 15 años' a 2
df['Antigüedad'] = df['Antigüedad'].map({'1 a 8 años': 1, '9 a 15 años': 2})


# Utilizar LabelEncoder para convertir los nombres de barrios en valores numéricos
# Esto es necesario para poder incluir la columna 'Barrio' en la matriz de correlación
label_encoder = LabelEncoder()
df['Barrio'] = label_encoder.fit_transform(df['Barrio'])


# Seleccionar únicamente las columnas con datos numéricos para la matriz de correlación
# Esto incluye columnas transformadas y originalmente numéricas
numeric_cols = df.select_dtypes(include=['number'])


# Calcular la matriz de correlación de Pearson
correlation_matrix = numeric_cols.corr(method='pearson')


# Crear un gráfico de calor (heatmap) para visualizar la matriz de correlación
# Utilizamos Seaborn para una visualización más intuitiva y estéticamente agradable
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")


# Añadir un título al gráfico para una mejor interpretación de los resultados
plt.title("Matriz de Correlación de Pearson")


# Mostrar el gráfico
plt.show()