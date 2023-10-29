

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('/content/adult.csv')
print(data)
data.describe()

data.isnull().sum()
import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
print(data.corr())

# plotting correlation heatmap
dataplot = sb.heatmap(data.corr(), cmap="YlGnBu", annot=True)

# displaying heatmap
mp.show()

from sklearn.preprocessing import OneHotEncoder

# Handle missing values
data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)

# Separate features and target
x = data.drop('income', axis=1)
y = data['income']

# Separate categorical and numerical columns
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

x_categorical = x[categorical_columns]
x_numerical = x[numerical_columns]

# Apply one-hot encoding to categorical features
encoder = OneHotEncoder()
x_categorical_encoded = encoder.fit_transform(x_categorical)

# Combine encoded categorical features with numerical features
import numpy as np
x_encoded = np.hstack((x_categorical_encoded.toarray(), x_numerical))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.3, random_state=1)
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest classifier object
clf = RandomForestClassifier(n_estimators=100, random_state=1)

# Train the classifier
clf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Predict on the test set
predictions = clf.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Generate classification report
report = classification_report(y_test, predictions)
print("Classification Report:\n", report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", conf_matrix)

