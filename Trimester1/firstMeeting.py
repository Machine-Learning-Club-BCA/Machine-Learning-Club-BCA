# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

weather_file_path = '/kaggle/input/predictingtheweather/Weather_Train.csv'

weather_data = pd.read_csv(weather_file_path)

from sklearn.tree import DecisionTreeRegressor
y = weather_data.temperature
feature_columns = ['minimum_temperature', 'maximum_temperature', 'dew_point', 'relative_humidity', 'heat_index', 'wind_speed', 'wind_direction', 'precipitation', 'precipitation_cover', 'visibility', 'cloud_cover', 'sea_level_pressure', 'latitude', 'longitude']
X = weather_data[feature_columns]

weather_data = DecisionTreeRegressor()
weather_data.fit(X, y)


predict_file_path = '/kaggle/input/predictingtheweather/Weather_Predict1.csv'
predict_data = pd.read_csv(predict_file_path)

val_X = predict_data[feature_columns]
val_y = predict_data.temperature

val_predictions = weather_data.predict(val_X)
# print(val_y)
print(val_predictions)
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)

print(val_mae)

# # Set up code checking
# from learntools.core import binder
# binder.bind(globals())
# from learntools.machine_learning.ex3 import *
# print("Setup good")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
