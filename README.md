# Decision Tree from Scratch / Árbol de decisión desde cero

![alt text](decision_tree-1.svg)

[English](#-english) / [Español](#-español)

---

## English

### Overview

This project builds **part of a decision tree classifier from scratch** (using only NumPy and pandas) and then compares the result with scikit-learn's `DecisionTreeClassifier`. The goal is to understand how a tree chooses its splits: which predictor, which threshold, and when to stop.

### Repository contents

| File | Description |
|---|---|
| `decision_tree_from_scratch.ipynb` | Notebook with the full step-by-step implementation |
| `two_classes.csv` | Toy dataset: 450 rows, two numeric predictors (`x1`, `x2`) and a binary label (`y`) |
| `README.md` | This file |

### Dataset

`two_classes.csv` is a synthetic dataset with no missing values:

| Column | Type | Range | Description |
|---|---|---|---|
| `x1` | int | 1 – 498 | First predictor |
| `x2` | int | 2 – 500 | Second predictor |
| `y` | int | 0 / 1 | Class label (230 samples of class 1, 220 of class 0, so nearly balanced) |

### Requirements

- Python 3.8+
- `numpy`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`
- Jupyter Notebook or JupyterLab

```bash
pip install numpy pandas scikit-learn seaborn matplotlib jupyter
jupyter notebook decision_tree_from_scratch.ipynb
```

### What was done, step by step

**0. Data exploration.** The data is loaded and plotted as a scatter plot (`x1` vs `x2`, coloured by class) to visually anticipate where the decision boundaries should fall.

**1. Root node: choosing the first split.**
- Every unique value of each predictor is tried as a candidate split.
- Splits are evaluated with the **Gini impurity**, `1 − Σ pᵢ²`, weighted by the size of each child node (`get_total_gini`).
- sklearn conventions are followed: the condition is `<= threshold`; points that satisfy it go to the **left** child, the rest go to the **right** child.
- The final threshold is the **midpoint** between the best value and the next higher unique value (`get_threshold`).
- Result: the best first split is on **`x2`** (best value 158, **threshold 159.5**).

**2. First split and decision regions.** Helper functions `get_split_labels` (majority class of each region) and `predict_class` (predicts over a `np.meshgrid`) are used with `plt.contourf` to draw the coloured decision regions. The data is then subset into `first_split_df` (`x2 <= 159.5`, 151 points).

**3. Second split.** The same procedure is applied to the left region. The best split is on **`x1`** (**threshold 137.5**), giving:
- Right child (`x1 > 137.5`): 111 points, all class 1 → **pure**.
- Left child (`x1 <= 137.5`): 40 points (35 of class 0, 5 of class 1) → **impure**.

**4. Stopping criterion.** Splitting stops on regions that are pure (a single class).

**5. Recursion on the impure region.** The left child of the second split is split again. The best split is on **`x2`** (**threshold 141.0**), producing two pure leaves. This completes the **left branch** of the tree.

**6. Comparison with scikit-learn.** A `DecisionTreeClassifier(max_depth=3)` is fitted on the same data and drawn with `tree.plot_tree`. Its left branch (`x2 <= 159.5` → `x1 <= 137.5` → `x2 <= 141.0`) matches the manual result exactly.

### Resulting left branch

```
x2 <= 159.5
├── x1 <= 137.5
│   ├── x2 <= 141.0  → class 0
│   └── x2 >  141.0  → class 1
└── x1 >  137.5      → class 1
```

### Key concepts covered

- Gini impurity and weighted impurity of a split
- Exhaustive search of candidate thresholds (unique values, midpoints)
- Recursive binary splitting and purity as a stopping criterion
- Visualising decision regions
- Validating a from-scratch implementation against a reference library

### Possible improvements

- Implement the full recursion (including the right branch) in a single `fit` function or class.
- Add other stopping criteria (`max_depth`, `min_samples_split`).
- Add a train/test split and evaluate accuracy.

---

## Español

### Descripción general

Este proyecto construye **parte de un clasificador de árbol de decisión desde cero** (usando solo NumPy y pandas) y después compara el resultado con el `DecisionTreeClassifier` de scikit-learn. El objetivo es entender cómo un árbol elige sus divisiones: qué variable, qué umbral y cuándo parar.

### Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `decision_tree_from_scratch.ipynb` | Notebook con la implementación completa paso a paso |
| `two_classes.csv` | Dataset de juguete: 450 filas, dos predictores numéricos (`x1`, `x2`) y una etiqueta binaria (`y`) |
| `README.md` | Este archivo |

### Dataset

`two_classes.csv` es un dataset sintético sin valores nulos:

| Columna | Tipo | Rango | Descripción |
|---|---|---|---|
| `x1` | int | 1 – 498 | Primer predictor |
| `x2` | int | 2 – 500 | Segundo predictor |
| `y` | int | 0 / 1 | Clase (230 muestras de clase 1 y 220 de clase 0, casi balanceado) |

### Requisitos

- Python 3.8+
- `numpy`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`
- Jupyter Notebook o JupyterLab

```bash
pip install numpy pandas scikit-learn seaborn matplotlib jupyter
jupyter notebook decision_tree_from_scratch.ipynb
```

### Qué se ha hecho, paso a paso

**0. Exploración de los datos.** Se cargan los datos y se dibuja un diagrama de dispersión (`x1` vs `x2`, coloreado por clase) para anticipar visualmente dónde deberían estar las fronteras de decisión.

**1. Nodo raíz: elegir la primera división.**
- Se prueba como candidato cada valor único de cada predictor.
- Las divisiones se evalúan con el **índice de Gini**, `1 − Σ pᵢ²`, ponderado por el tamaño de cada nodo hijo (`get_total_gini`).
- Se siguen las convenciones de sklearn: la condición es `<= umbral`; los puntos que la cumplen van al hijo **izquierdo** y el resto al **derecho**.
- El umbral final es el **punto medio** entre el mejor valor y el siguiente valor único mayor (`get_threshold`).
- Resultado: la mejor primera división es sobre **`x2`** (mejor valor 158, **umbral 159,5**).

**2. Primera división y regiones de decisión.** Se usan las funciones auxiliares `get_split_labels` (clase mayoritaria de cada región) y `predict_class` (predice sobre un `np.meshgrid`) junto con `plt.contourf` para dibujar las regiones de decisión coloreadas. Después se filtran los datos en `first_split_df` (`x2 <= 159,5`, 151 puntos).

**3. Segunda división.** Se aplica el mismo procedimiento a la región izquierda. La mejor división es sobre **`x1`** (**umbral 137,5**), con lo que se obtiene:
- Hijo derecho (`x1 > 137,5`): 111 puntos, todos de clase 1 → **puro**.
- Hijo izquierdo (`x1 <= 137,5`): 40 puntos (35 de clase 0 y 5 de clase 1) → **impuro**.

**4. Criterio de parada.** Se deja de dividir en las regiones puras (una sola clase).

**5. Recursión sobre la región impura.** El hijo izquierdo de la segunda división se vuelve a dividir. La mejor división es sobre **`x2`** (**umbral 141,0**) y genera dos hojas puras. Con esto se completa la **rama izquierda** del árbol.

**6. Comparación con scikit-learn.** Se entrena un `DecisionTreeClassifier(max_depth=3)` con los mismos datos y se dibuja con `tree.plot_tree`. Su rama izquierda (`x2 <= 159,5` → `x1 <= 137,5` → `x2 <= 141,0`) coincide exactamente con el resultado manual.

### Rama izquierda resultante

```
x2 <= 159.5
├── x1 <= 137.5
│   ├── x2 <= 141.0  → clase 0
│   └── x2 >  141.0  → clase 1
└── x1 >  137.5      → clase 1
```

### Conceptos clave

- Impureza de Gini e impureza ponderada de una división
- Búsqueda exhaustiva de umbrales candidatos (valores únicos, puntos medios)
- División binaria recursiva y pureza como criterio de parada
- Visualización de regiones de decisión
- Validación de una implementación propia frente a una librería de referencia

### Posibles mejoras

- Implementar la recursión completa (incluida la rama derecha) en una función o clase `fit`.
- Añadir otros criterios de parada (`max_depth`, `min_samples_split`).
- Añadir una partición train/test y evaluar la precisión.
