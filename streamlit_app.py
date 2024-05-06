import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the dataset
@st.cache_resource
def load_data():
    data = pd.read_csv('cpdata.csv') 
    return data

data = load_data()

# Define features and target
features = ['temperature', 'humidity', 'ph', 'rainfall']
target = 'label'

# Preprocess the data
X = data[features]
y = data[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
svm_model = SVC()
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Save models
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(dt_model, 'models/dt_model.pkl')

# Streamlit UI
def main():
    st.title('Crop Prediction')

    # Sidebar for user input
    st.sidebar.title('Input Data')
    input_data = {}

    # Collect user input
    for feature in features:
        input_data[feature] = st.sidebar.text_input(f'Enter {feature}:')

    if st.sidebar.button('Predict'):
        # Preprocess the input data
        input_df = pd.DataFrame(input_data, index=[0])
        input_df_scaled = scaler.transform(input_df)

        # Make predictions
        svm_prediction = svm_model.predict(input_df_scaled)
        rf_prediction = rf_model.predict(input_df_scaled)
        dt_prediction = dt_model.predict(input_df_scaled)

        # Display predictions
        st.subheader('Predictions')
        st.write(f'SVM Model Prediction: {svm_prediction}')
        st.write(f'Random Forest Model Prediction: {rf_prediction}')
        st.write(f'Decision Tree Model Prediction: {dt_prediction}')

if __name__ == '__main__':
    main()
