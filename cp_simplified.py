# Import essential libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  # Consider classification model
from sklearn.metrics import accuracy_score

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


 # Predicted values from model 
ah=tp['Air Humidity']
atemp=tp['Air Temp']
shum=tp['Soil Humidity']
pH=tp['Soil pH']
rain=tp['Rainfall']


# Evaluate model accuracy on unseen data
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
print("Model accuracy:", accuracy)
