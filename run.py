import subprocess
import sys
import os

def main():
    print("🚀 Iris Flower Classification Project Setup")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('iris_model.pkl') and not os.path.exists('model.pkl'):
        print("⚠️ No model found. Training a new model...")
        try:
            # Import and run training
            from train_and_save_model import train_and_save_model
            train_and_save_model()
        except ImportError:
            print("❌ Could not import training module.")
            print("📝 Please run: python3 train_and_save_model.py")
            return
    else:
        print("✅ Model file found.")
    
    # Run the web app
    print("\n🌐 Starting web application...")
    print("📡 Open http://localhost:5000 in your browser")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    try:
        import app
        app.model_data = app.load_or_create_model()
        app.app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()