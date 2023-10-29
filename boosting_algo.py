

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('adult.csv')
print(data)

data.isnull().sum()

# Replace '?' with NaN in the dataset
data.replace('?', pd.NA, inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Split the data into training and testing sets
X = data.drop("income", axis=1)
y = data["income"]

categorical_columns = X.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
X[categorical_columns] = X[categorical_columns].apply(label_encoder.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# Create the AdaBoost classifier
ada_boost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)

# Fit the classifier to the training data
ada_boost_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ada_boost_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("The Accuracy for boosting algo is :", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Assuming you already have the y_test and y_pred values from your AdaBoost classifier
confusion_matrix = confusion_matrix(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create a title for the plot with accuracy score
title = f'Confusion Matrix - Score: {round(accuracy, 2)}'

# Create the ConfusionMatrixDisplay
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

# Plot the confusion matrix with the specified title plt.figure(figsize=(8, 6))
cm_display.plot(cmap='Oranges_r', values_format='d')
plt.title(title, size=15)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print the accuracy score
print("Accuracy Score:", accuracy)

print("Classification Report:\n", report)