# TFMCOD

Modelo Predictivo de Precios de la Vivienda en Estrato 6 de Chapinero, Bogotá: Aplicación de Técnicas de Machine Learning

Descripción: 
Este proyecto incluye scripts en Python y notebooks de Jupyter para la predicción de precios de inmuebles en Chapinero, Bogotá, utilizando técnicas de Machine Learning y análisis de correlación.

Estructura del Repositorio:
- Datos: Carpeta donde está ubicada la base de datos utilizada en los modelos.
- Modelos: Contiene los modelos entrenados en formato .pkl, el código de Python y el notebook de jupyther documentados, que incluyen la transformación de variables, modelado, evaluación y despliegue de los mismos a través de una interfaz gráfica.
- Correlación: Contiene un notebook y un script para calcular la matriz de correlación de Pearson de todos las variables analizadas.

Instrucciones de Instalación y Ejecución:
1. Clonar el repositorio desde GitHub.
2. Asegurarse de tener la base de datos en la carpeta 'Datos' del repositorio.
3. Instalar las dependencias necesarias utilizando el archivo 'requirements.txt' ejecutando `pip install -r requirements.txt` en su entorno de Python.
4. Ejecutar el script de Python 'Código.py' o el notebook 'CódigoTFM.ipynb' desde la carpeta 'Modelos' para entrenar los modelos y acceder a la interfaz gráfica.
5. Para analizar la correlación, abra el notebook o el script en la carpeta 'Correlación' y ejecútelo. Asegúrese de que la ruta de la base de datos esté correctamente configurada.

Dependencias (ver 'requirements.txt'):
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pickle
- tkinter
- jupyter (opcional para notebooks)

Notas Adicionales:
- Los scripts y el notebook utilizan rutas relativas para acceder a la base de datos. Mantener la estructura del repositorio como está para evitar problemas con las rutas.
- Si experimentas problemas al cargar la base de datos o los modelos, verificar que las rutas relativas en los scripts y el notebook sean consistentes con la estructura de tu directorio local.
