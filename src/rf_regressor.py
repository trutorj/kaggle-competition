# Import your libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV   
from sklearn.externals import joblib
####################################################################3

# Load the prepared datasets
train = pd.read_csv('data/train_prepared.csv'), index_col=0)
test = pd.read_csv('data/test_prepared.csv'), index_col=0)

# Define X and y to fit the train model
X = train.drop(columns=["price", "y"])
y = train.price

# Split the samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)

#########################################################################
#                        RANDOM FOREST                                  #
#########################################################################

# Create a dictionary to store the parameters that will be optimized
params = {'n_estimators' : [1000],
          'max_depth': [10,30,50]
                        }
tuning = GridSearchCV(estimator=RandomForestRegressor(criterion='mse',
                                                      min_samples_split=2,
                                                      min_samples_leaf=1,
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto',
                                                      max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0,
                                                      min_impurity_split=None,
                                                      bootstrap=True,
                                                      oob_score=False,
                                                      n_jobs=None,
                                                      random_state=None,
                                                      verbose=0,
                                                      warm_start=False,
                                                      ccp_alpha=0.0,
                                                      max_samples=None),
                      param_grid = params,
                      verbose = 500,
                      n_jobs = -1,
                      cv = 5,
                      scoring = 'neg_root_mean_squared_error')
tuning.fit(X_train, y_train)

# Save the best model
joblib.dump(tuning.best_estimator_, '../output/rf.pkl'), compress = 1)

# Load the model
best_model = joblib.load('../output/rf.pkl')

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
pd.DataFrame(pred_test, columns=["price"]).to_csv('output/RandomForestRegressor.csv'))
