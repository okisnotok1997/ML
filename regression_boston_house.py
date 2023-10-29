

# Commented out IPython magic to ensure Python compatibility.
# Importing the libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# Displaying the data
df = pd.read_csv('Boston Dataset.csv')
df.head(3)

df.drop(columns=['Unnamed: 0'], axis=0, inplace=True)
df.head()

# Statistical info
df.describe()

# datatype info
df.info()

# Checking for null values
df.isnull().sum()

# Creating box plots for attributes
# box plots are used for indentifying outliners
# An outlier is an observation that lies an abnormal distance
# from other values in a random sample from a population

fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))  #7*2 = 14, since 14 attributes
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index +=1

# Hyper parameter tunning to display graph properly
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# The dot's in the box plot's are outliners
# By observing the below figure we can see that
# CRIM, ZM, B have many outliners
# To deal with outliners we can use min-max transformation or ignore the outliners by deleting the rows or dropping the colums

# Create dist plot
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))  #7*2 = 14, since 14 attributes
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index +=1

# Hyper parameter tunning to display graph properly
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# Left skewed - CRIM, ZN, DIS
# Right skewed - AGE, B
# Double bell - INDUS, RAD, TAX
# Complete uniform distribution - RM, LSTAT

# CRIM, ZN, TAX, B -> Min max normalization wil be done

# Min-max normalization
cols = ['crim', 'zn', 'tax', 'black']
for col in cols:
    # Find minimum and maximum of that column
    minimum = min(df[col])
    maximum = max(df[col])
    df[col] = (df[col] - minimum) / (maximum - minimum)

fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))  #7*2 = 14, since 14 attributes
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index +=1

# Hyper parameter tunning to display graph properly
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# standardization
from sklearn import preprocessing
scalar = preprocessing.StandardScaler()

# fit the data
scaled_cols = scalar.fit_transform(df[cols])
scaled_cols = pd.DataFrame(scaled_cols, columns=cols)
scaled_cols.head()

for col in cols:
    df[col] = scaled_cols[col]

fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))  #7*2 = 14, since 14 attributes
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index +=1

# Hyper parameter tunning to display graph properly
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# Coorelation matrix
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True)

sns.regplot(y=df['medv'], x=df['rm'])

sns.regplot(y=df['medv'], x=df['lstat'])

# input split
X = df.drop(columns=['medv', 'rad'], axis=1)
y = df['medv']

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model.fit(X_train, y_train)

# predict the training set
pred = model.predict(X_test)

# perform cross-validation
cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
cv_score = np.abs(np.mean(cv_score))

print("MSE:", mean_squared_error(y_test, pred))
print('CV Score:', cv_score)