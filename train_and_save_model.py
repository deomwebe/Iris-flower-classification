import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    """Train a model and save it to a file"""
    
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model trained with accuracy: {accuracy:.2%}")
    
    # Prepare data to save
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names,
        'accuracy': accuracy
    }
    
    # Save the model
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Model saved to 'iris_model.pkl'")
    
    # Also save a simple version for the web app
    simple_model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'target_names': ['setosa', 'versicolor', 'virginica']
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(simple_model_data, f)
    
    print("✅ Simple model saved to 'model.pkl'")
    print("\nYou can now run the web app with: python3 app.py")
    
    return model_data

if __name__ == "__main__":
    train_and_save_model()