# Kaggle Competition
#### *Encontrar el mejor modelo para predecir el precio de los diamantes*

## 1. Introducción

El objetivo de este proyecto consiste en encontrar el modelo de regresión que mejor estime el precio de un diamente basado en las siguientes características:

* **Carat**: es el peso de la piedra.

* **Color**: el grado de ausencia de color.

* **Cut**: las proporciones del diamante.

* **Claridad**: la presencia o ausencia de inclusiones en el diamante.

* **Depth**: altura del diamente.

* **Table**: anchura de la cara superior.

* **x, y, z**: métricas dimensionales.

Para ello, contamos con un dataset compuesto de  40345 entradas que relacionan estas características y el precio alcanzado por el diamante y de los diferentes modelos de machine learning de las librerías SciKit Learning y H2O.

## 2. Materiales y métodos

### Análisis de los datos

El primer paso fue ver los datos, su tipo, su estructura, la presencia de nulos y las relaciones entre las variables. En la siguiente tabla se muestra el resumen del análisis exploratorio de los datos:

|       |        carat |         cut |       color |     clarity |       depth |       table |           x |           y |            z |    price |
|:------|-------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|-------------:|---------:|
| count | 40345        | 40345       | 40345       | 40345       | 40345       | 40345       | 40345       | 40345       | 40345        | 40345    |
| mean  |     0.795652 |     3.55385 |     4.41324 |     4.05802 |    61.7504  |    57.4603  |     5.72611 |     5.73022 |     3.53514  |  3924.09 |
| std   |     0.470806 |     1.02692 |     1.69524 |     1.64864 |     1.42422 |     2.23533 |     1.11869 |     1.14858 |     0.693662 |  3982    |
| min   |     0.2      |     1       |     1       |     1       |    43       |    43       |     0       |     0       |     0        |   326    |
| 25%   |     0.4      |     3       |     3       |     3       |    61       |    56       |     4.71    |     4.72    |     2.91     |   948    |
| 50%   |     0.7      |     3       |     4       |     4       |    61.8     |    57       |     5.69    |     5.71    |     3.52     |  2395    |
| 75%   |     1.04     |     4       |     6       |     5       |    62.5     |    59       |     6.54    |     6.53    |     4.03     |  5313    |
| max   |     4.01     |     5       |     7       |     8       |    79       |    95       |    10.02    |    58.9     |     8.06     | 18818    |

Para analizar las correlaciones entre las distintas variables, se hizo un heatmap:
![alt text](img/heatmap.png)

Se observa lo siguiente:

- *carat* es la variable que mejor se correlaciona con el precio.

- *x*,*y*,*z* están muy correlacionadas entre sí y con el precio.

- El resto de variables apenas tienen correlación con el precio.

Gráficos de dispersión entre precio y el resto de variables
![alt text](img/scatterplots.png)
De la gráfica y de la bibliografía consultada se extrae que entre *price* y *carat* hay una relación exponencial. De hecho, si 

### Extracción de las variables

Como paso previo al entrenamiento del modelo, y viendo los resultados del análisis anterior, por un lado, se decidió eliminar las variables *table* y *depth* por su baja correlación con *price*; por otro se crearon 3 nuevas variables al hacer el cociente entre *cut*, *color* y *clarity* y *carat* con el objetivo de no perder dicha información.

### Análisis de los modelos

#### Modelos de Skicit Learn

Para decidir qué modelo escoger, se seleccionaron los siguientes modelos de la librería Skicit Learn y se ejecutaron para ver cuáles tenían mejores métricas:

* Linear Regression

* Lasso Regression

* Ridge Regression

* Support Vector Regression (con kernel polinomial)

* Decission Tree Regressor

* Gradient Boost Regressor

* Random Forest Regressor

Los dos modelos con los mejores resultados se seleccionaron para una segunda fase en la que se ajustaron algunos de sus parámetros mediante la herramienta `GridSearchCV`.

#### Modelo de H2O AutoML

Por último, también se hizo una búsqueda del mejor modelo a través de H2O AutoML, con una selección de 20 modelos y un límite de una hora.

## 3. RESULTADOS

#### Modelos de Skicit Learn

En la siguiente tabla se muestran las métricas conseguidas por cada uno de los modelos con sus parámetros *de fábrica*.

|         |   LinearRegression |   LassoRegression |   RidgeRegression |   SupportVectorRegression |   DecissionTreeRegressor |   GradientBoostRegressor |   RandomForest |
|:--------|-------------------:|------------------:|------------------:|--------------------------:|-------------------------:|-------------------------:|---------------:|
| mae     |         845.095    |        846.49     |        845.616    |                704.493    |               366.667    |               372.75     |     278.533    |
| Rsquare |           0.875335 |          0.875353 |          0.875312 |                 -0.844836 |                 0.964027 |                 0.972573 |       0.980231 |
| rmse    |        1386.84     |       1386.74     |       1386.97     |               5334.99     |               744.976    |               650.496    |     552.264    |


Para el ajuste más fino, se seleccionaron el Gradient Boost Regressor y el Random Forest Regressor. Los parámetros con los que se obtuvieron los mejores resultados se muestran a continuación (solo se muestran los que se ajustaron, el resto se dejaron por defecto):

* Gradient Boosting Regressor:
    - loss='lad'
    - learning_rate=0.05
    - max_depth=10
    - n_estimators=500

* Random Forest Regressor
    - n_estimators = 1000
    - max_depth = 50

Las métricas obtenidas para cada modelo

| Metrica  |Gradient Boosting Regressor| Random Forest Regressor|
|:---------|--------------------------:|-----------------------:|
|mae       |                   290.9415|                281.3454|
|Rsquare   |                     0.9792|                  0.9787|
|rmse      |                     576.88|                  576.19|

#### Modelo de H2OML

Finalmente, el mejor modelo encontrado por H2OML fue un modelo ensamblado, StackedEnsemble_AllModels_AutoML_20200511_162219, con el que se obtuvieron las siguientes métricas:
| Metrica               | Valor    |
|:----------------------|---------:|
|mean_residual_deviance |    290786|
|rmse                   |   539.246|
|mse                    |    290786|
|mae                    |   274.409|
|rmsle                  |    0.0990|

