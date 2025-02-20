import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Generación de datos más realistas
np.random.seed(42)
n_samples = 1000
complejidad = np.random.randint(1, 6, size=n_samples)
tiempo = np.random.randint(1, 11, size=n_samples)
prioridad = np.where((complejidad + tiempo) <= 5, 'Baja',
                    np.where((complejidad + tiempo) <= 8, 'Media',
                             np.where((complejidad + tiempo) <= 10, 'Alta', 'Crítica')))

datos = {
    'Complejidad': complejidad,
    'Tiempo': tiempo,
    'Prioridad': prioridad
}

df = pd.DataFrame(datos)

# Balanceo de clases sin el warning
df_balanced = (
    df.groupby('Prioridad', group_keys=False)  # Evita incluir la columna 'Prioridad' en la operación
    .apply(lambda x: x.sample(n=250, replace=True))
    .reset_index(drop=True)
)

# Codificación de etiquetas
le = LabelEncoder()
df_balanced['Prioridad'] = le.fit_transform(df_balanced['Prioridad'])

# Separar características y etiquetas
X = df_balanced[['Complejidad', 'Tiempo']]
y = df_balanced['Prioridad']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Optimización de hiperparámetros con GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

modelo = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Mejor modelo
mejor_modelo = grid_search.best_estimator_

# Evaluación con validación cruzada
cv_scores = cross_val_score(mejor_modelo, X, y, cv=5, scoring='accuracy')
print(f"Exactitud media en validación cruzada: {cv_scores.mean():.2f}")

# Evaluación en el conjunto de prueba
y_pred = mejor_modelo.predict(X_test)
print("Exactitud en prueba:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=mejor_modelo.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.show()

# Predicción con formato correcto
nueva_tarea_df = pd.DataFrame([[4, 7]], columns=['Complejidad', 'Tiempo'])
prediccion = mejor_modelo.predict(nueva_tarea_df)
print(f"Predicción para 'Optimizar código', complejidad 4, tiempo 7: {le.inverse_transform(prediccion)[0]}")

# Visualización del árbol
def graficar_arbol(modelo, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(modelo, feature_names=feature_names, class_names=le.classes_, filled=True, rounded=True)
    plt.show()

graficar_arbol(mejor_modelo, X.columns)

