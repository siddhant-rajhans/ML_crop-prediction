# Import essential libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  # Consider classification model
from sklearn.metrics import accuracy_score
#from firebase import firebase
#import time  # for sleep function

# Define crop labels (modify these based on your actual labels)
crops=['wheat','mungbean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']

# Load and prepare data
data = pd.read_csv('cpdata.csv')

# Handle categorical labels (modify based on your data)
if not pd.api.types.is_numeric_dtype(data['label']):
    data = pd.get_dummies(data, columns=['label'], drop_first=True)  # One-hot encoding

X = data.iloc[:, :-1]  # Features (all columns except the last, which is the label)
y = data.iloc[:, -1]  # Labels (last column)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model (consider classification model)
model = DecisionTreeClassifier()  # Change to a classification model if needed
model.fit(X_train, y_train)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
print("Model accuracy:", accuracy)

# Define retry parameters (adjust as needed)
max_retries = 5
retry_delay = 2  # Seconds

# Fetch real-time data from Firebase  ----- Not fixed yet
firebase = firebase.FirebaseApplication('https://ml-crop-prediction-97361-default-rtdb.firebaseio.com/data')
for attempt in range(max_retries + 1):
  try:
    realtime_data = firebase.get('/Realtime', None)
    break  # Exit loop if data is retrieved successfully
  except Exception as e:
    print(f"Error fetching live data (attempt {attempt}/{max_retries}): {e}")
    if attempt < max_retries:
      print(f"Waiting {retry_delay} seconds before retrying...")
      time.sleep(retry_delay)
    else:
      print("Reached maximum retries, exiting...")
      exit(1)  # Exit with an error code

# Check for missing sensor data (optional)
if realtime_data is None:
  print("Error: No data found in Firebase!")

  # Existing code to check for missing sensor data within realtime_data

  # Handle missing data (e.g., use default values or interpolate)

# Extract live features with error handling
live_features = []
for sensor in ["Air Humidity", "Air Temp", "Soil pH", "Rainfall"]:
  value = realtime_data.get(sensor)
  if value is None:
    print(f"Warning")
