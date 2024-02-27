import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors

# Read the dataset
df = pd.read_csv('breast-cancer-wisconsin.data')
# Replace missing values represented by '?' with a large negative number
df.replace('?', -99999, inplace=True)
# Drop the 'id' column as it's not relevant for prediction
df.drop(['id'], axis=1, inplace=True)

# Prepare feature and target matrices
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Initialize and train the K-Nearest Neighbors classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# Evaluate the model's accuracy on the test set
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Predict a new sample
example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(f"Prediction: {prediction}")
