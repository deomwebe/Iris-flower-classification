from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Try to load the model, or create a default one if it doesn't exist
def load_or_create_model():
    """Load existing model or create a new one"""
    
    model_files = ['iris_model.pkl', 'model.pkl']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                print(f"✅ Model loaded from {model_file}")
                return model_data
            except Exception as e:
                print(f"⚠️ Error loading {model_file}: {e}")
    
    # If no model file exists, train a simple model
    print("⚠️ No model file found. Training a new model...")
    return train_simple_model()

def train_simple_model():
    """Train a simple model if no saved model exists"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Simple train/test split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    # Create model data
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'target_names': ['setosa', 'versicolor', 'virginica']
    }
    
    # Save it for future use
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ New model trained and saved to 'model.pkl'")
    return model_data

# Load the model when the app starts
model_data = load_or_create_model()
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
target_names = model_data['target_names']

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/api/info')
def api_info():
    """Return information about the model"""
    return jsonify({
        'model_type': type(model).__name__,
        'feature_names': feature_names,
        'target_names': target_names.tolist() if hasattr(target_names, 'tolist') else target_names,
        'features_count': len(feature_names),
        'classes_count': len(target_names)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
            
        # Extract features
        features = [
            float(data.get('sepal_length', 0)),
            float(data.get('sepal_width', 0)),
            float(data.get('petal_length', 0)),
            float(data.get('petal_width', 0))
        ]
        
        # Validate inputs
        for i, value in enumerate(features):
            if value <= 0:
                return jsonify({
                    'error': f'Invalid value for {feature_names[i]}: {value}. Must be positive.'
                }), 400
        
        # Prepare features array
        features_array = np.array([features])
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get species name
        species_name = target_names[prediction]
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'species': species_name,
            'probabilities': {
                target_names[0]: float(probabilities[0]),
                target_names[1]: float(probabilities[1]),
                target_names[2]: float(probabilities[2])
            },
            'confidence': float(np.max(probabilities)),
            'measurements': {
                'sepal_length': features[0],
                'sepal_width': features[1],
                'petal_length': features[2],
                'petal_width': features[3]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/example/<int:example_id>')
def get_example(example_id):
    """Get example measurements"""
    examples = [
        {'name': 'Setosa', 'values': [5.1, 3.5, 1.4, 0.2]},
        {'name': 'Versicolor', 'values': [6.0, 2.7, 4.5, 1.5]},
        {'name': 'Virginica', 'values': [7.7, 3.0, 6.1, 2.3]}
    ]
    
    if 0 <= example_id < len(examples):
        return jsonify(examples[example_id])
    else:
        return jsonify({'error': 'Example not found'}), 404

if __name__ == '__main__':
    print("\n🌺 Iris Flower Classification Web App")
    print("=" * 40)
    print(f"Model: {type(model).__name__}")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Classes: {', '.join(target_names)}")
    print("\n🌐 Starting web server...")
    print("📡 Open http://localhost:5000 in your browser")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)