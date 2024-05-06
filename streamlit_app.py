import streamlit as st
import pandas as pd
import numpy as np  # Add this line for NumPy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('cpdata.csv')
    if not pd.api.types.is_numeric_dtype(data['label']):
        data = pd.get_dummies(data, columns=['label'], drop_first=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Data loaded successfully!')

# Define crop labels
crops = ['wheat', 'mungbean', 'Tea', 'millet', 'maize', 'lentil', 'jute', 'cofee',
         'cotton', 'ground nut', 'peas', 'rubber', 'sugarcane', 'tobacco',
         'kidney beans', 'moth beans', 'coconut', 'blackgram', 'adzuki beans',
         'pigeon peas', 'chick peas', 'banana', 'grapes', 'apple', 'mango',
         'muskmelon', 'orange', 'papaya', 'watermelon', 'pomegranate']

# Split data into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train models
svm_model = SVC()
rf_model = RandomForestClassifier()
dt_model = DecisionTreeClassifier()

svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Evaluate model performance
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))

# Create Streamlit app
st.title('Crop Prediction')

# Display model accuracies
st.write('Model Accuracies:')
st.write(f'SVM Model Accuracy: {svm_accuracy}')
st.write(f'Random Forest Model Accuracy: {rf_accuracy}')
st.write(f'Decision Tree Model Accuracy: {dt_accuracy}')

# Prediction using randomly selected data
random_index = np.random.randint(0, len(X_test))
random_data = X_test[random_index].reshape(1, -1)
random_data_scaled = scaler.transform(random_data)

svm_prediction = svm_model.predict(random_data_scaled)[0]
rf_prediction = rf_model.predict(random_data_scaled)[0]
dt_prediction = dt_model.predict(random_data_scaled)[0]

st.write('Randomly Selected Data Prediction:')
st.write(f'SVM Model Prediction: {crops[int(svm_prediction)]}')  # Fix the warning by converting to int
st.write(f'Random Forest Model Prediction: {crops[int(rf_prediction)]}')  # Fix the warning by converting to int
st.write(f'Decision Tree Model Prediction: {crops[int(dt_prediction)]}')  # Fix the warning by converting to int
