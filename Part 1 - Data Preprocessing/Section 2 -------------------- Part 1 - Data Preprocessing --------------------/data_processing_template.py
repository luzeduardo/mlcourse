import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

dataset = pd.read_csv('Data.csv')
#[:,:-1] means: get all value rows and ignore last collumn
X = dataset.iloc[:, :-1].values
#[:,:3] means: get all value rows for the third collumn
y = dataset.iloc[:, 3].values
# print X
#used to fit values where it is empty inside the matrix
#using default strategy of "mean" that is the average of values from collumn on axis x that means collumn, not row that is 1
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#all lines and collumns 1 and 2 that has empty values
imputer_data = imputer.fit(X[:, 1:3])
#this will replace inside the dataset values from imputer in the empty values
X[:, 1:3] = imputer_data.transform(X[:, 1:3])


#categoring data
label_encoder_x = LabelEncoder()
#this will generate a list containing country names as numbers for each one, but this is not the final solution because
#the machine learning will assume that there is an priority based on the numbers
#so i will use onehotencoder to generate a new grid containing for each country the bit turned on
#(france, spain, germany)
# (1,0,0) is paris
# (0,1,0) is spain this will category data without some priority based on numbers
X[:, 0] = label_encoder_x.fit_transform(X[:, 0])
#specify with collumn I will use to create category based on label, used collumn 0 that is country
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
y = label_encoder_x.fit_transform(y)

# print X
# print y

#splitting the dataset into the training set and the data set
from sklearn.model_selection import train_test_split
#defining 20% of data to test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


##Featuring scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print X

