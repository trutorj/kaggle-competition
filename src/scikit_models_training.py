# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV   
from sklearn.externals import joblib
####################################################################################################
# Load prepared datasets
train = pd.read_csv('data/train_prepared.csv'), index_col=0)
test = pd.read_csv('data/test_prepared.csv'), index_col=0)

# Define X and y to fit the train model
X = train.drop(columns=["price", "y"])
y = train.price

# Split the samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)

#################################################################
#                  SciKitLearn MODELS                           #
#################################################################
# Create a dictionary to store the selected models:
models = {
    "LinearRegression": LinearRegression(),
    "LassoRegression": Lasso(alpha=1),
    "RidgeRegression": Ridge(alpha=1),
    "SupportVectorRegression": SVR(kernel='poly'),
    "DecissionTreeRegressor": DecisionTreeRegressor(),
    "GradientBoostRegressor": GradientBoostingRegressor()
    "RandomForest": RandomForestRegressor()
}

# Run all the models to inspect their performance and choose the best ones
metrics = pd.DataFrame()
for model_name, model in models.items():
    print(f"Training model: {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred)
    Rsquare = model.score(X_test,y_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    metricas = pd.DataFrame([mae,Rsquare, rmse],
                            index=["mae", "Rsquare", "rmse"],
                            columns=[model_name])
    metrics = pd.concat([metrics, metricas], axis=1)

# Print the metrics as dataframe
metrics


