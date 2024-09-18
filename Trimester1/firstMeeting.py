# IMPORTANT: Copy this code and put it in the kaggle notebook to make it.



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
