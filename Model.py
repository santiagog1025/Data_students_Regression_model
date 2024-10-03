
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("Student_Performance.csv")
print(df.head(3))
print('\n\r')
df.describe()

df.info()
print(df.shape)
print('\n\r')
print(df["Extracurricular Activities"].value_counts())

df.isnull().sum()

# Contar las actividades extracurriculares
activity_counts = df['Extracurricular Activities'].value_counts()
# Crear una figura con subgrillas
fig, axs = plt.subplots(3, 2, figsize=(10, 9))

# Histograma para Hours Studied
sns.histplot(df['Hours Studied'], bins=30, kde=True, ax=axs[0, 0])
axs[0, 0].set_title('Distribución de Horas Estudiadas')
axs[0, 0].set_xlabel('Horas Estudiadas')
axs[0, 0].set_ylabel('Frecuencia')

# Histograma para Previous Scores
sns.histplot(df['Previous Scores'], bins=30, kde=True, ax=axs[0, 1])
axs[0, 1].set_title('Distribución de Puntuaciones Anteriores')
axs[0, 1].set_xlabel('Puntuaciones Anteriores')
axs[0, 1].set_ylabel('Frecuencia')

# Histograma para Sleep Hours
sns.histplot(df['Sleep Hours'], bins=30, kde=True, ax=axs[1, 0])
axs[1, 0].set_title('Distribución de Horas de Sueño')
axs[1, 0].set_xlabel('Horas de Sueño')
axs[1, 0].set_ylabel('Frecuencia')

# Histograma para Sample Question Papers Practiced
sns.histplot(df['Sample Question Papers Practiced'], bins=30, kde=True, ax=axs[1, 1])
axs[1, 1].set_title('Distribución de Papeles de Preguntas Practicados')
axs[1, 1].set_xlabel('Papeles de Preguntas Practicados')
axs[1, 1].set_ylabel('Frecuencia')

# Histograma para Performance Index
sns.histplot(df['Performance Index'], bins=30, kde=True, ax=axs[2, 0])
axs[2, 0].set_title('Distribución del Índice de Rendimiento')
axs[2, 0].set_xlabel('Índice de Rendimiento')
axs[2, 0].set_ylabel('Frecuencia')

sns.barplot(x=activity_counts.index, y=activity_counts.values)
axs[2, 1].set_title('Distribución de Actividades Extracurriculares')
axs[2, 1].set_xlabel('Actividades Extracurriculares')
axs[2, 1].set_ylabel('Frecuencia')

# Ajustar el layout
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x='Hours Studied', y='Performance Index', data=df, color='darkgreen')
plt.title('Hours Studied vs Performance Index')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.show()

plt.figure(figsize=(8, 6))
sns.lineplot(x='Previous Scores', y='Performance Index', data=df, color='darkred')
plt.title('Previous Scores vs Performance Index')
plt.xlabel('Previous Scores')
plt.ylabel('Performance Index')
plt.show()

pie_data = df.groupby('Extracurricular Activities')['Performance Index'].mean()
plt.figure(figsize=(8, 6))
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=['tomato', 'darkred'], startangle=90, explode=(0.05, 0))  # explode ile dilimlerden birini ayırabilirsiniz
plt.title('Performance Index based on Extracurricular Activities')
plt.show()

# Matriz de Correlación
df_corr = df.select_dtypes(include=['float64','int64'])
plt.figure(figsize=(10, 6))
correlation_matrix = df_corr.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='OrRd', linewidths=0.5)
plt.title('Matriz de Correlación entre Variables')
plt.show()


# Generar un preprocesador que organice las columnas para tener el mejor modelo posible
# Definir las columnas categóricas y numéricas
categorical_col = ['Extracurricular Activities']  # Nombre de la columna categórica
numerical_cols = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']  # Nombres de las columnas numéricas

# Preprocesador que estandariza las columnas numéricas y codifica la categórica
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Escalar columnas numéricas
        ('cat', OneHotEncoder(), categorical_col)  # Codificar la columna categórica
    ]
)
X = df[numerical_cols + categorical_col]
y = df['Performance Index']
# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Supongamos que la columna categórica es la columna 3 (índice 2)

# Crear un pipeline que incluya el preprocesamiento y el modelo
pipeline_linear = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Primero aplicar el preprocesamiento
    ('model', LinearRegression())     # Luego, aplicar el modelo de regresión lineal
])

# Ajustar el modelo
pipeline_linear.fit(X_train, y_train)

"""#### **Beneficios del Pipeline**
* Modularidad: Los pasos del preprocesamiento y modelado se aplican en secuencia sin la necesidad de realizar cada uno por separado.
* Reutilización: Puedes aplicar el mismo pipeline en diferentes conjuntos de datos sin necesidad de realizar el preprocesamiento manualmente cada vez.
* Cross-Validation y Grid Search: Facilita la evaluación de modelos y la búsqueda de hiperparámetros óptimos mediante validación cruzada o grid search, sin tener que preocuparte por aplicar el preprocesamiento de forma incorrecta en cada conjunto de datos.
"""

# Hacer predicciones
y_pred_linear = pipeline_linear.predict(X_test)

# Evaluar el modelo
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2 = r2_score(y_test, y_pred_linear)

print(f'Error Cuadrático Medio (MSE): {mse_linear}')
print(f'Raiz del Error Cuadrático Medio (RMSE): {np.sqrt(mse_linear)}')
print(f'Coeficiente de Determinación (R²): {r2}')



# Crear un pipeline que incluya el preprocesamiento y la regresión polinómica
pipeline_poli = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Primero aplicar el preprocesamiento
    ('poly', PolynomialFeatures()),   # Agregar la transformación polinómica
    ('model', LinearRegression())     # Modelo de regresión lineal
])

# Definir el rango de grados para GridSearch
param_grid = {
    'poly__degree': [1, 2, 3, 4, 5]  # Grados del polinomio a probar
}

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(pipeline_poli, param_grid, cv=5, scoring='neg_mean_squared_error')

# Ajustar el modelo utilizando GridSearchCV
grid_search.fit(X_train, y_train)

# Hacer predicciones
y_pred_poly = grid_search.predict(X_test)

# Calcular el MSE y R²
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2 = r2_score(y_test, y_pred_poly)

# Imprimir los resultados
print(f'Mejor grado del polinomio: {grid_search.best_params_}')
print(f'Error Cuadrático Medio (MSE): {mse_poly}')
print(f'Raíz del error Cuadrático Medio (RMSE): {np.sqrt(mse_poly)}')
print(f'Coeficiente de Determinación (R²): {r2}')



# Crear una figura y establecer el tamaño
plt.figure(figsize=(14, 6))

# Gráfico de regresión lineal
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, color='darkblue', label='Linear Regression')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='white', linestyle='--')  # Línea y = x
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Regresión Lineal: Valores Reales vs Predicciones')
plt.legend()

# Gráfico de regresión polinómica
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_poly, color='darkred', label='Polynomial Regression')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')  # Línea y = x
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Regresión Polinómica: Valores Reales vs Predicciones')
plt.legend()

# Mostrar las gráficas
plt.tight_layout()
plt.show()
print(f'Error Cuadrático Medio (MSE) Regresión Lineal: {mse_linear}')
print(f'Error Cuadrático Medio (MSE) Regresión Polinómica: {mse_poly}')

