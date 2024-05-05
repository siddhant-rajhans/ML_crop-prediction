# Import essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  # Consider classification model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Define crop labels (modify these based on your actual labels)
crops = ['wheat', 'mungbean', 'Tea', 'millet', 'maize', 'lentil', 'jute', 'cofee',
        'cotton', 'ground nut', 'peas', 'rubber', 'sugarcane', 'tobacco',
        'kidney beans', 'moth beans', 'coconut', 'blackgram', 'adzuki beans',
        'pigeon peas', 'chick peas', 'banana', 'grapes', 'apple', 'mango',
        'muskmelon', 'orange', 'papaya', 'watermelon', 'pomegranate']

# Load and prepare data (assuming 'cpdata.csv' has your data)
data = pd.read_csv('cpdata.csv')

# Handle categorical labels (modify if your labels are text-based)
if not pd.api.types.is_numeric_dtype(data['label']):
    data = pd.get_dummies(data, columns=['label'], drop_first=True)  # One-hot encoding
    # Alternatively, you can use pd.get_dummies(data, columns=['label'])
    # convert text into numerical features using one-hot encoding. 

# Separate features (all columns except the last) and labels (last column)
# .iloc refers to position
# .loc refers to index
# [start : end]
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]  # Labels

# Split data into training and testing sets (70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale features (improves model performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train a decision tree model (can be replaced with other models)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Import libraries for SVM and Random Forest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Create SVM and Random Forest models
svm_model = SVC()
rf_model = RandomForestClassifier()

# Fit the models
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Evaluate model performance on test data
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test)) * 100
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test)) * 100

print("SVM model accuracy:", svm_accuracy)
print("Random Forest model accuracy:", rf_accuracy)



# Calculate precision, recall, F1, and confusion matrix
svm_precision = precision_score(y_test, svm_model.predict(X_test))
svm_recall = recall_score(y_test, svm_model.predict(X_test))
svm_f1 = f1_score(y_test, svm_model.predict(X_test))
svm_cm = confusion_matrix(y_test, svm_model.predict(X_test))

rf_precision = precision_score(y_test, rf_model.predict(X_test))
rf_recall = recall_score(y_test, rf_model.predict(X_test))
rf_f1 = f1_score(y_test, rf_model.predict(X_test))
rf_cm = confusion_matrix(y_test, rf_model.predict(X_test))

print("SVM model precision:", svm_precision)
print("SVM model recall:", svm_recall)
print("SVM model F1:", svm_f1)
print("SVM model confusion matrix:\n", svm_cm)

print("Random Forest model precision:", rf_precision)
print("Random Forest model recall:", rf_recall)
print("Random Forest model F1:", rf_f1)
print("Random Forest model confusion matrix:\n", rf_cm)



# Evaluate model accuracy on unseen data
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
print("Model accuracy:", accuracy)


# # **Display data from CSV**

# # Show the first row of the data (assuming informative features)
# print("\nData preview (first row):")
# display(data.head(1))  # Use display() for Jupyter Notebook output

# # **Prediction using randomly selected data**

# # Select a random sample from the test set (replace 1 with your desired sample size)
# random_index = numpy.random.randint(0, len(X_test) - 1)
# random_data = X_test[random_index].reshape(1, -1)

# # Scale the random data using the same scaler
# random_data_scaled = scaler.transform(random_data)

# # Make prediction on the random data
# prediction = model.predict(random_data_scaled)[0]  # Get the first element

# # Print the predicted crop based on the index
# print("\nPredicted crop:", crops[prediction])

# **Display a sample data point**
print("\nSample data point:")
print(data.head(1))  # Display the first row

# **Display data after one-hot encoding (assuming you have a single label column)**
print("\nThe data present in one row of the dataset after one-hot encoding:")
encoded_data = pd.get_dummies(data.iloc[0], columns=['label'], drop_first=True)
print(pd.concat([data.iloc[0, :-1], encoded_data], axis=1))  # Combine features and encoded labels

# **Prediction using randomly selected data**

# Select a random sample from the test set (replace 1 with your desired sample size)
random_index = np.random.randint(0, len(X_test) - 1)
random_data = X_test[random_index].reshape(1, -1)

# Scale the random data using the same scaler
random_data_scaled = scaler.transform(random_data)

# Make prediction on the random data
prediction = model.predict(random_data_scaled)[0]  # Get the first element

# Print the predicted crop based on the index
print("\nThe predicted crop is:", crops[prediction])
