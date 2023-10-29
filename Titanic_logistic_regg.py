
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# load the data from csv file to Pandas DataFrame
df = pd.read_csv('train.csv')

# printing the first 5 rows of the dataframe
df.head()

# number of rows and Columns
df.shape

# getting some informations about the data
df.info()

# check the number of missing values in each column
df.isnull().sum()

# drop the "Cabin" column from the dataframe
df = df.drop(columns='Cabin', axis=1)

# replacing the missing values in "Age" column with mean value
df['Age'].fillna(df['Age'].mean(), inplace=True)

# finding the mode value of "Embarked" column
print(df['Embarked'].mode())

print(df['Embarked'].mode()[0])

# replacing the missing values in "Embarked" column with mode value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# check the number of missing values in each column
df.isnull().sum()

"""Data Analysis"""

# getting some statistical measures about the data
df.describe()

# finding the number of people survived and not survived
df['Survived'].value_counts()

# making a count plot for "Survived" column
sns.countplot(x='Survived', data=df)

df['Sex'].value_counts()

# making a count plot for "Sex" column
sns.countplot(x='Sex', data=df)

# number of survivors Gender wise
sns.countplot(x='Sex', hue='Survived', data=df)

# making a count plot for "Pclass" column
sns.countplot(x='Pclass', data=df)

sns.countplot(x='Pclass', hue='Survived', data=df)

df['Sex'].value_counts()

df['Embarked'].value_counts()

# converting categorical Columns
df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

df.head()

X = df.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = df['Survived']

X.head(3)

Y.head(3)

#Splitting the data into training data & Test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

model = LogisticRegression()

# training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)

X_train_prediction

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)

X_test_prediction

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

print("Confusion matrix :-")
sns.heatmap(confusion_matrix(Y_test, X_test_prediction), annot=True)

from sklearn.metrics import classification_report
print(classification_report(Y_test, X_test_prediction))