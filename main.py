import pandas as pd
import numpy as np
import calendar
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("train.csv")

train_labels = train.count

del train['registered']
del train['casual']
del train['count']

# Extracting new features from datetime in training set
train.datetime = train.datetime.apply(pd.to_datetime)
train['day'] = train.datetime.apply(lambda date: date.day)
train['month'] = train.datetime.apply(lambda date: date.month)
train['year'] = train.datetime.apply(lambda date: date.year)
train['weekday'] = train.datetime.apply(lambda date: calendar.weekday(date.year, date.month, date.day))
train['hour'] = train.datetime.apply(lambda date: date.hour)

del train['datetime']

