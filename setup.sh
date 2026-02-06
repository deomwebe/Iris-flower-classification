#!/bin/bash

# setup.sh - Complete setup script

echo "🚀 Setting up Iris Flower Classification Project"
echo "================================================"

# Create directory structure
mkdir -p templates

echo "📁 Created directory structure"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Install dependencies
echo "📦 Installing dependencies..."
pip install flask scikit-learn numpy pandas --quiet

# Create app.py if it doesn't exist
if [ ! -f "app.py" ]; then
    echo "📝 Creating app.py..."
    cat > app.py << 'EOF'
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and train model
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
target_names = iris.target_names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'success': True,
            'species': target_names[prediction],
            'probabilities': {
                target_names[0]: float(probabilities[0]),
                target_names[1]: float(probabilities[1]),
                target_names[2]: float(probabilities[2])
            },
            'confidence': float(np.max(probabilities))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/example/<int:example_id>')
def get_example(example_id):
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
    print("🌺 Iris Flower Classification Web App")
    print("🌐 Server starting... Open http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
EOF
fi

# Check if index.html exists
if [ ! -f "templates/index.html" ]; then
    echo "📄 Creating index.html template..."
    echo "Please add your HTML code to templates/index.html"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 To run the application:"
echo "   1. Add your HTML code to templates/index.html"
echo "   2. Run: python3 app.py"
echo "   3. Open: http://localhost:5000"
echo ""
echo "🛠️  For help, check the troubleshooting guide."