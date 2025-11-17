Repositorio: https://github.com/Juan0204/PruebaDS

# Imputación de dimensiones y peso de productos a partir de especificaciones

Este proyecto construye un pipeline en Python para:

1. Leer un archivo CSV con información de productos.
2. Parsear la columna de `specifications` en formatos muy diversos (JSON, listas de dicts, texto, etc.).
3. Extraer **dimensiones** (largo, ancho, alto) y **peso** en unidades homogéneas (pulgadas y libras).
4. Generar features adicionales (texto, conteos, features geométricas).
5. Entrenar modelos de regresión (Random Forest) para completar valores faltantes.
6. Entregar un `DataFrame` final con columnas numéricas imputadas (`*_final`) listas para análisis o consumo aguas arriba.

El script principal del proyecto es `main_script.py`, que contiene todo el pipeline de punta a punta.

# Explicación detallada

El código implementa un pipeline completo en Python que toma un CSV de productos, parsea la columna specifications (que puede venir como JSON, listas de diccionarios, texto libre, etc.) y la normaliza a un diccionario _spec_dict por fila; a partir de ahí extrae las claves estándar de dimensiones y peso ("Assembled Product Dimensions (L x W x H)" y "Assembled Product Weight"), convierte el texto de dimensiones a tres columnas numéricas (length_in, width_in, height_in) en pulgadas y el texto de peso a weight_lb en libras, manejando múltiples unidades (kg, g, oz, lb). Luego construye un DataFrame result con columnas de contexto (categoría, marca, nombre, descripción, tags, etc.) más las columnas físicas, enriquece con features adicionales: versiones string de columnas tipo lista y conteos (por ejemplo n_review_tags), y genera features geométricas derivadas de largo y ancho (area_in2, razón L/W, mínimo, máximo, diagonal, densidades peso/área, etc. y sus logs). Con ese dataset enriquecido entrena cuatro modelos de regresión (Random Forest, envueltos en un pipeline con imputación numérica/categórica y OneHotEncoder) para predecir cada variable física (length_in, width_in, height_in, weight_lb), usando como predictores las otras dimensiones/peso, los conteos numéricos y las columnas categóricas de contexto; cada modelo se evalúa con holdout y validación cruzada, e incluso calcula importancias por permutación. Finalmente, usa esos modelos para imputar los valores faltantes de las cuatro variables físicas, sobrescribiendo los NaN con predicciones y creando columnas finales (length_in_final, width_in_final, height_in_final, weight_lb_final). La función build_full_pipeline orquesta todo: lectura del CSV, parseo, cálculo físico, construcción de la vista, enriquecimiento de features, entrenamiento de modelos e imputación; main simplemente llama a ese pipeline, imprime un resumen y devuelve el DataFrame final junto con el diccionario de modelos entrenados.

Este bloque implementa una interfaz de línea de comandos (CLI) para ejecutar el pipeline completo desde la terminal. Utiliza argparse para recibir dos argumentos obligatorios: --input, que indica la ruta del archivo CSV de entrada, y --output, que define la ruta donde se guardará el archivo resultante con las dimensiones y el peso imputados. Dentro de main(), estos parámetros se convierten en rutas tipo Path, se carga el CSV indicado, se ejecuta el pipeline build_full_pipeline y finalmente se guarda el DataFrame procesado en el archivo de salida. El bloque if __name__ == "__main__": asegura que el script se ejecute como programa independiente desde consola, permitiendo usarlo así: python main_script.py --input productos.csv --output productos_imputados.csv.

# Build de la imagen
Dataset Prueba Técnica - market-products.csv
cli_script.py
main_script.py
requirements.txt
Dockerfile

docker build -t mi_pipeline:latest .

docker run --rm -v "$PWD":/data mi_pipeline:latest \
  --input "/data/Dataset Prueba Técnica - market-products.csv" \
  --output "/data/output_enriquecido.csv"


# ¿Por qué se utilizó Random Forest para la imputación de dimensiones y peso?

Para la tarea de imputar dimensiones y peso de productos se eligió Random Forest Regressor porque ofrece un equilibrio sólido entre rendimiento, estabilidad y facilidad de uso en datasets con características muy heterogéneas. A diferencia de modelos lineales tradicionales, Random Forest no asume relaciones lineales entre las variables y maneja sin problemas las múltiples interacciones no lineales entre atributos como categorías, texto codificado, conteos, densidades y medidas físicas. También es robusto ante valores ruidosos, escalas distintas y columnas con distribuciones sesgadas, características propias de este tipo de datos de catálogo de productos. Frente a modelos más complejos como XGBoost o redes neuronales, Random Forest evita sobreajuste con menos ajustes de hiperparámetros, entrena más rápido, requiere menos ingeniería de features y funciona bien incluso con cantidades moderadas de datos. Además, sus predicciones son estables incluso si algunas dimensiones están completamente ausentes y deben inferirse a partir de contexto semántico (categoría, marca, tags, descripción). Finalmente, este modelo permite usar permutation importance, lo que facilita interpretar qué variables influyen más en cada predicción y justificar la imputación generada. En conjunto, estas razones hacen de Random Forest una elección práctica, interpretable y eficaz para completar automáticamente valores faltantes en dimensiones y peso.