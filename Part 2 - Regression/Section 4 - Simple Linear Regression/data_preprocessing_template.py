# Data Preprocessing Template

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fit linear regression  to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


#visualising the training set
# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years')
# plt.ylabel('Salary')
# plt.show()

#visualising the test set
plt.scatter(X_test, y_test, color='black')
plt.plot(X_train, regressor.predict(X_train), color='yellow')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show()

stop = 1