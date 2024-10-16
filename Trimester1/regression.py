student_file_path = 'student-mat.csv'
student_data = pd.read_csv(student_file_path, sep=';')

y= student_data.G1
# student_data.describe()

student_features = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'goout', 'Dalc','health', 'absences']

X = student_data[student_features]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
model = RandomForestRegressor(random_state=1)
model.fit(train_X, train_y)
student_preds = model.predict(val_X)
print(mean_absolute_error(val_y, student_preds))
print(student_preds[0:5])
print(val_y.head())
