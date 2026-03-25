

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  
import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\VS Code\PYTHON\machine_learning\car_price.csv")

X = df.drop('price',axis = 1)
Y = df['price']

X_one_encode = pd.get_dummies(X,columns = ['model','transmission','fuelType'],drop_first = True)

X_one_encode = X_one_encode.astype(int)

from sklearn.preprocessing import LabelEncoder

columns = ['model', 'transmission', 'fuelType']

Xlable = X.copy()  
label_encoders = {}

for col in columns:
    le = LabelEncoder()
    Xlable[col] = le.fit_transform(Xlable[col].astype(str))  # Convert to string in case of nulls
    label_encoders[col] = le

from sklearn.preprocessing import StandardScaler

numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
scaler = StandardScaler()
X_one_encode[numerical_cols] = scaler.fit_transform(X_one_encode[numerical_cols])

Xlable[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg',
    'engineSize']] = scaler.fit_transform(Xlable[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg',
    'engineSize']])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X_one_encode, Y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
# print(y_pred
r2 = r2_score(y_test,y_pred)

print(r2)