import pandas as pd
import numpy as np
import calendar
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import preprocessing, pipeline, linear_model
from sklearn.model_selection import train_test_split

'''
SUBMISSION FILE NAME
'''
submission_filename = 'RandomForestRegressor-notemp_improved_submission.csv'

'''
MODEL NAMES: XGBRegressor, GradientBoostingRegressor, RandomForestRegressor, Lasso, LinearRegression
'''
model_name = 'RandomForestRegressor'


# Extracting new features from datetime in training set
def feature_engineering(dataset):
    dataset.datetime = dataset.datetime.apply(pd.to_datetime)
    dataset['day'] = dataset.datetime.apply(lambda date: date.day)
    dataset['month'] = dataset.datetime.apply(lambda date: date.month)
    dataset['year'] = dataset.datetime.apply(lambda date: date.year)
    dataset['weekday'] = dataset.datetime.apply(lambda date: calendar.weekday(date.year, date.month, date.day))
    dataset['hour'] = dataset.datetime.apply(lambda date: date.hour)
    del dataset['datetime']

    # delete not used columns
    if  'count' in dataset:
        del dataset['count']
    if 'casual' in dataset:
        del dataset['casual']
    if 'registered' in dataset:
        del dataset['registered']

    return dataset


# treats each type of feature (categorical, binary and numerical)
def transform_feature_type(dataset):
    binary_columns = ['holiday', 'workingday']
    binary_indices = np.array([(column in binary_columns) for column in dataset.columns], dtype=bool)

    categorical_columns = ['season', 'weather', 'month', 'year', 'weekday']
    categorical_indices = np.array([(column in categorical_columns) for column in dataset.columns], dtype=bool)

    numeric_columns = ['temp', 'humidity', 'windspeed', 'hour', 'day']
    numeric_indices = np.array([(column in numeric_columns) for column in dataset.columns], dtype=bool)

    return binary_indices, categorical_indices, numeric_indices


# returns a transformer list with the preprocessing for each type of feature
# this transformer list will be used by the estimator during training and testing
def populate_transformer(dataset):
    binary_indices, categorical_indices, numeric_indices = transform_feature_type(dataset)
    return [
        # binary features makes no preprocessing.. only constructs a transformer
        ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_indices])),
        # continuous data... standard normalization with mean zero
        ('numeric_variables_processing', pipeline.Pipeline(
            steps=[('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_indices])),
                   ('scaling', preprocessing.StandardScaler(with_mean=0))])),
        # discrete values for categorical data are preprocessed to one hot enconder
        ('categorical_variables_processing', pipeline.Pipeline(
            steps=[ ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_indices])),
                    ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown='ignore'))
        ])),
    ]


# Root Mean Squared Logarithmic Error
def rmsle(y_true, y_pred):
    log_pred = np.nan_to_num(np.array([np.log(v + 1) for v in y_pred])) # Replace nan with zero and inf with finite numbers.
    log_true = np.nan_to_num(np.array([np.log(v + 1) for v in y_true])) # Replace nan with zero and inf with finite numbers.
    calc = (log_pred - log_true) ** 2
    return np.sqrt(np.mean(calc))


x_train = pd.read_csv("train.csv")
x_train["count"] = np.log(x_train["count"])
y_train = x_train['count']

x_train = feature_engineering(x_train)

# split data into train and test sets
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=42)

# fit model in training data
if model_name == 'XGBRegressor':
    model = XGBRegressor(max_depth=25, learning_rate=0.1, n_estimators=130)
elif model_name == 'GradientBoostingRegressor':
    model = GradientBoostingRegressor(max_depth = 25, learning_rate=0.1, n_estimators=130)
elif model_name == 'RandomForestRegressor':
    model = RandomForestRegressor(max_depth = 25, n_estimators = 130, random_state = 0)
elif model_name == 'Lasso':
    model = linear_model.Lasso(max_iter = 2000)
elif model_name == 'LinearRegression':
    model = linear_model.LinearRegression()

# creates a tranformer list
transformer = populate_transformer(x_train)

# instanciates the estimator using the pipeline for feature processing and the XGBR model
estimator = pipeline.Pipeline(steps = [
    ('feature_processing', pipeline.FeatureUnion(transformer_list=transformer)),
    ('model_fitting', model)
    ]
)

# fits training dta
estimator.fit(x_train, y_train)

# make predictions for test data
y_pred = estimator.predict(x_test)
#y_pred = [round(value) for value in y_pred]

# calculates rmlse
print rmsle(y_test, y_pred)

# makes prediction over the real test dataset from kaggle
x_real_test = pd.read_csv("test.csv")

# maintain datetime values from test set
x_real_test_datetime = x_real_test['datetime']

# applies feature engineering to dataset
x_real_test = feature_engineering(x_real_test)

# makes predictions over real test set
y_real_test_pred = estimator.predict(x_real_test)

# writes output file for submission
submission = pd.DataFrame({"datetime": x_real_test_datetime,
                           "count": [np.exp(x) for x in y_real_test_pred]},
                           columns=['datetime', 'count'])

print submission.head(20)

submission.to_csv('submissions/' + submission_filename, index=False)
