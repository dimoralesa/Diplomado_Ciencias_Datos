# Connection
Crear archivo de configuración para la conexión a la base de datos. 
```
touch connection/config.py
```
## Ejemplo config.py
```
server = "serverAddr" 
database = "db"  
username = "user"
password = "password" 
```

# Model
Correr el archivo model.py. Puede tardar hasta dos horas. Esto creará los archivos model/jc_model_results_future.json y model/jc_model_results_test.json. Test es el resultado del entrenamiento y future es el resultado de las predicciones. 

# Notas
La carga de los datos de ventas de la base de datos puede tardarse hasta 5 minutos. 