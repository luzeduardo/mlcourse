import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('Data.csv')
#[:,:-1] means: get all value rows and ignore last collumn
X = dataset.iloc[:, :-1].values
#[:,:3] means: get all value rows for the third collumn
y = dataset.iloc[:, 3].values
print X
#used to fit values where it is empty inside the matrix
#using default strategy of "mean" that is the average of values from collumn on axis x that means collumn, not row that is 1
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#all lines and collumns 1 and 2 that has empty values
imputer_data = imputer.fit(X[:, 1:3])
#this will replace inside the dataset values from imputer in the empty values
X[:, 1:3] = imputer_data.transform(X[:, 1:3])
print X
