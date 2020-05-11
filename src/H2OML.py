import h2o
from h2o.automl import H2OAutoML
h2o.init()
###################################################################################################
# Load prepared datasets
train= h2o.import_file('data/train_prepared.csv')
test= h2o.import_file('data/test_prepared.csv')

# Define X and y to train the models
x = train.columns
y = "price"
x.remove(y)

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb = automl.get_leaderboard(aml, extra_columns = 'ALL')

# Make the prediction with test data
preds = aml.leader.predict(test)

# Export as data frame
df = preds.as_data_frame()

# Export as csv
pd.DataFrame(df).to_csv('output/h20pred.csv')