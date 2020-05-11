# Import your libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.externals import joblib
##############################################################

# Load the prepared datasets
train = pd.read_csv('data/train_prepared.csv'), index_col=0)
test = pd.read_csv('data/test_prepared.csv'), index_col=0)

# Define X and y to fit the train model
X = train.drop(columns=["price", "y"])
y = train.price

# Split the samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)

#########################################################################
#                        GRADIENT BOOSTING                              #
#########################################################################

# Create a dictionary to store the parameters that will be optimized
params = {'loss': ["ls", "lad", "huber", "quantile"],
          'n_estimators' : [50, 100, 500, 1000],
          'max_depth': [10, 20, 30, 40],
          'learning_rate': [0.05, 0.1, 0.2]
                   }
tuning = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,
                                                          subsample=1.0,
                                                          criterion='friedman_mse',
                                                          min_samples_split=2,
                                                          min_samples_leaf=1,
                                                          min_weight_fraction_leaf=0.0,
                                                          min_impurity_decrease=0.0,
                                                          min_impurity_split=None,
                                                          init=None,
                                                          random_state=None,
                                                          max_features=None,
                                                          alpha=0.9,
                                                          verbose=0,
                                                          max_leaf_nodes=None,
                                                          warm_start=False,
                                                          presort='deprecated',
                                                          validation_fraction=0.1,
                                                          n_iter_no_change=None,
                                                          tol=0.0001,
                                                          ccp_alpha=0.0),
                      param_grid = params,
                      verbose = 500,
                      n_jobs = -1,
                      cv = 5,
                      scoring = 'neg_root_mean_squared_error')
tuning.fit(X_train, y_train)

# Save the best model
joblib.dump(tuning.best_estimator_, '../output/gbr.pkl'), compress = 1)

# Load the model
best_model = joblib.load('../output/gbr.pkl')

# Check it
best_model

# Fit the model, calculate the metrics and save it as csv
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
Rsquare = model.score(X_test,y_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
metricas = pd.DataFrame([mae,Rsquare, rmse],
                            index=["mae", "Rsquare", "rmse"])
# Predict with test data
pred_test = model.predict(test)
# Export as csv
pd.DataFrame(pred_test, columns=["price"]).to_csv('output/GradientBoostRegressor.csv'))
