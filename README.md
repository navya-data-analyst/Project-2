Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
Importing the Dataset
df = pd.read_csv("/content/car data.csv")
Data Inseception
df.head()
df.tail()
df.shape
df.info()
df.desribe()
Data Cleaning
df.isnull().sum()
df.duplicated().sum()
set the feature and Target Variable
x = df.drop(['Selling_price','car_Name'],axis=1)
y = df['Selling_price']
Split the Data
X = pd.get_dummies(X, drop_first=True)
X.head()
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
Feature Scaling
categorical_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)
Fit the Model
X_encoded = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
Find the Prediction Data
y_pred = model.predict(X_test)
Evaluation
print("R2_score:",r2_score(y_test,y_pred))
Visualization
import matplotlib.pyplot as plt
plt.scatter(y_train, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")
plt.show()
