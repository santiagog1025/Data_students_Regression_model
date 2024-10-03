# Análisis de Rendimiento Estudiantil con Regresión Lineal y Polinómica

Este proyecto tiene como objetivo evaluar cómo diferentes factores influyen en el rendimiento académico de los estudiantes utilizando modelos de regresión lineal y polinómica. A continuación se explica el código, su estructura, y cómo interpretar los resultados, haciendo especial énfasis en cuándo utilizar regresión lineal o polinómica.

## Descripción del Código

### Bibliotecas Utilizadas
Se utilizan las siguientes bibliotecas de Python para manipular, analizar y visualizar los datos, así como para construir los modelos:
- **pandas**: Manipulación y análisis de datos.
- **numpy**: Operaciones matemáticas.
- **matplotlib** y **seaborn**: Visualización de datos.
- **scikit-learn**: Preprocesamiento de datos y creación de los modelos de regresión.

### Análisis Exploratorio de Datos (EDA)
El archivo `Student_Performance.csv` se carga y se realiza un análisis exploratorio para comprender mejor la estructura de los datos:
- **Visualización preliminar**: Se muestran las primeras filas de los datos y se obtienen estadísticas descriptivas para conocer mejor las características del conjunto de datos.
- **Estructura de los datos**: Se explora la naturaleza de las columnas y los tipos de datos, lo que ayuda a identificar posibles problemas y entender el contexto de los datos.
- **Distribuciones**: Se analizan las distribuciones de variables clave como horas estudiadas, horas de sueño y puntuaciones previas, utilizando gráficos de densidad e histogramas.

### Visualización de Actividades Extracurriculares
Se genera un gráfico de barras para visualizar la distribución de las actividades extracurriculares y su relación con el rendimiento académico. Además, se examinan otras relaciones mediante gráficos de dispersión y lineales para entender cómo diferentes factores afectan el índice de rendimiento.

### Preprocesamiento de los Datos
Para preparar los datos para el modelado, se realizan los siguientes pasos:
- **Escalado de variables numéricas**: Se estandarizan las variables numéricas para asegurar que todas las características tengan una escala similar, lo que mejora la convergencia de los modelos.
- **Codificación de variables categóricas**: Las variables categóricas, como las actividades extracurriculares, se transforman en una representación numérica adecuada para los modelos de regresión.

### Modelos de Regresión

#### Regresión Lineal
El modelo de regresión lineal se utiliza para modelar la relación entre las variables predictoras y la variable dependiente (Índice de Rendimiento) asumiendo una relación lineal entre ellas. Se implementa un pipeline que automatiza el preprocesamiento y el ajuste del modelo de regresión lineal.

**Métricas de evaluación**:
- **MSE (Error Cuadrático Medio)**: Mide el error promedio al cuadrado entre los valores reales y las predicciones.
- **R² (Coeficiente de Determinación)**: Indica qué proporción de la variabilidad de la variable dependiente es explicada por el modelo.

#### Regresión Polinómica
Cuando la relación entre las variables no es lineal, la regresión polinómica puede capturar mejor las complejidades de los datos al agregar términos polinómicos. Se utiliza un pipeline similar al de la regresión lineal, pero incluyendo la transformación polinómica. Para determinar el mejor grado del polinomio, se utilizan técnicas de validación cruzada.

**Métricas de evaluación**:
- **MSE**: Error promedio de las predicciones.
- **R²**: Proporción de la variabilidad explicada por el modelo.

### Comparación entre Regresión Lineal y Polinómica
El proyecto compara los modelos de regresión lineal y polinómica utilizando las mismas métricas de evaluación. Se generan gráficos que muestran cómo los modelos predicen en comparación con los valores reales.

## Cuándo Utilizar Regresión Lineal vs. Polinómica
- **Regresión Lineal**: Se debe utilizar cuando las variables predictoras y la variable dependiente presentan una relación aproximadamente lineal. Este modelo es más simple y fácil de interpretar.
  
- **Regresión Polinómica**: Se recomienda cuando la relación entre las variables es más compleja y no puede ser capturada por una simple línea recta. Sin embargo, si el grado del polinomio es bajo, el modelo polinómico puede ser equivalente a una regresión lineal.

En resumen:
- Si el **MSE** es significativamente menor y el **R²** es mayor en el modelo polinómico, significa que hay una relación no lineal que está siendo capturada.
- Si los resultados de la regresión polinómica son similares a los de la regresión lineal (por ejemplo, para un grado 1), entonces ambas relaciones pueden ser lineales.

### Resultados
- **Regresión Lineal**: MSE y R² obtenidos del modelo.
- **Regresión Polinómica**: Mejor grado del polinomio, MSE y R² obtenidos del modelo.

### Conclusión
Este proyecto muestra cómo las técnicas de regresión pueden utilizarse para analizar datos educativos. La decisión entre usar regresión lineal o polinómica depende de la naturaleza de la relación entre las variables predictoras y la variable dependiente. Mientras que la regresión lineal es más simple, la polinómica puede capturar relaciones más complejas, ofreciendo mayor flexibilidad cuando se ajusta a datos con patrones no lineales.
