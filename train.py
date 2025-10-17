import pandas as pd
import numpy as np
import os # Import the OS module to handle directories
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib 

# --- 1. Data Loading and Inspection ---
print("--- 1. Data Loading ---")
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
target_names = iris.target_names

# --- 2. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. Model Training ---
print("--- 3. Training KNN Model ---")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
print("Model training successful.")

# --- 4. Model Evaluation ---
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# --- 5. Saving the Model (With Directory Check) ---
model_directory = 'model'
model_filename = f'{model_directory}/model.pkl'

# Create the directory if it does not exist
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    print(f"Created directory: {model_directory}")

# Save the trained model 
joblib.dump(knn_model, model_filename)
print(f"✅ Model successfully saved as '{model_filename}'")