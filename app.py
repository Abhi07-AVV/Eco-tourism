"""
Flask Backend API for Eco-Tourism Climate Risk Prediction
Optimized for Render.com deployment with enhanced error handling
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global variables to store loaded models and processors
models = {}
scalers = {}
encoders = {}
feature_names = {}

def load_models_and_processors():
    """Load all trained models and data processors with enhanced error handling"""
    global models, scalers, encoders, feature_names
    
    try:
        print("Starting model loading process...")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        
        # Check if model files exist
        model_files = [
            'best_regression_model_linear.pkl',
            'best_classification_model_logistic.pkl',
            'regression_scaler.pkl',
            'classification_scaler.pkl',
            'regression_encoders.pkl',
            'classification_encoders.pkl',
            'feature_names.json'
        ]
        
        missing_files = [f for f in model_files if not os.path.exists(f)]
        if missing_files:
            print(f"Missing model files: {missing_files}")
            print("Please ensure all .pkl files are present in the repository.")
            return False
        
        print("All required files found. Loading models...")
        
        # Load models
        models['regression'] = joblib.load('best_regression_model_linear.pkl')
        models['classification'] = joblib.load('best_classification_model_logistic.pkl')
        print("✅ Models loaded successfully")
        
        # Load scalers
        scalers['regression'] = joblib.load('regression_scaler.pkl')
        scalers['classification'] = joblib.load('classification_scaler.pkl')
        print("✅ Scalers loaded successfully")
        
        # Load encoders
        encoders['regression'] = joblib.load('regression_encoders.pkl')
        encoders['classification'] = joblib.load('classification_encoders.pkl')
        print("✅ Encoders loaded successfully")
        
        # Load feature names
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        print("✅ Feature names loaded successfully")
        
        # Validate the loaded data
        print(f"Loaded models: {list(models.keys())}")
        print(f"Loaded scalers: {list(scalers.keys())}")
        print(f"Loaded encoders: {list(encoders.keys())}")
        print(f"Loaded feature_names: {list(feature_names.keys())}")
        
        print("All models and processors loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_input_data(data, task_type='regression'):
    """Preprocess input data with enhanced error handling"""
    try:
        print(f"Preprocessing data for task_type: {task_type}")
        
        # Validate that required objects are loaded
        if not models:
            raise Exception("Models not loaded. Please check server logs.")
        
        if task_type not in encoders:
            raise Exception(f"No encoders found for task_type: {task_type}. Available: {list(encoders.keys())}")
        
        if task_type not in scalers:
            raise Exception(f"No scalers found for task_type: {task_type}. Available: {list(scalers.keys())}")
        
        if task_type not in feature_names:
            raise Exception(f"No feature_names found for task_type: {task_type}. Available: {list(feature_names.keys())}")
        
        # Create DataFrame from input
        df = pd.DataFrame([data])
        print(f"Input data columns: {list(df.columns)}")
        
        # Handle categorical encoding
        categorical_features = ['Vegetation_Type', 'Soil_Type', 'Country']
        for feature in categorical_features:
            if feature in df.columns and feature in encoders[task_type]:
                encoder = encoders[task_type][feature]
                # Handle unknown categories by assigning 0
                df[feature] = df[feature].apply(
                    lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else 0
                )
                print(f"Encoded feature: {feature}")
        
        # Handle boolean column
        if 'Protected_Area_Status' in df.columns:
            df['Protected_Area_Status'] = df['Protected_Area_Status'].astype(int)
            print("Converted Protected_Area_Status to int")
        
        # Select only feature columns needed for the model
        feature_cols = feature_names[f'{task_type}_features']
        print(f"Required features: {feature_cols}")
        
        # Check if all required features are present
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise Exception(f"Missing required features: {missing_features}")
        
        df_features = df[feature_cols]
        
        # Scale features
        scaled_features = scalers[task_type].transform(df_features)
        print(f"Successfully preprocessed data. Shape: {scaled_features.shape}")
        
        return scaled_features
        
    except Exception as e:
        print(f"Error in preprocess_input_data: {str(e)}")
        raise Exception(f"Error preprocessing data: {str(e)}")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions with enhanced error handling"""
    try:
        print("Received prediction request")
        
        # Check if models are loaded
        if not models:
            return jsonify({
                'error': 'Models not loaded. Please check server configuration.',
                'success': False
            }), 500
        
        # Get input data from request
        input_data = request.json
        print(f"Input data keys: {list(input_data.keys()) if input_data else 'No data'}")
        
        if not input_data:
            return jsonify({'error': 'No input data provided', 'success': False}), 400
        
        # Validate required fields
        required_fields = [
            'Latitude', 'Longitude', 'Vegetation_Type', 'Biodiversity_Index',
            'Protected_Area_Status', 'Elevation_m', 'Slope_Degree', 'Soil_Type',
            'Air_Quality_Index', 'Average_Temperature_C', 'Tourist_Attractions',
            'Accessibility_Score', 'Tourist_Capacity_Limit'
        ]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}', 'success': False}), 400
        
        # Add default values for fields needed by the model
        defaults = {
            'Country': 'USA',
            'Flood_Risk_Index': 0.3,
            'Drought_Risk_Index': 0.3,
            'Temperature_C': input_data.get('Average_Temperature_C', 20.0),
            'Annual_Rainfall_mm': 1000.0,
            'Soil_Erosion_Risk': 0.2,
            'Current_Tourist_Count': input_data.get('Tourist_Capacity_Limit', 500) * 0.6,
            'Human_Activity_Index': 0.4,
            'Conservation_Investment_USD': 100000.0,
            'Climate_Risk_Score': 0.4
        }
        
        for key, default_value in defaults.items():
            if key not in input_data:
                input_data[key] = default_value
        
        print("Processing predictions...")
        
        # Preprocess data for both models
        regression_data = preprocess_input_data(input_data, 'regression')
        classification_data = preprocess_input_data(input_data, 'classification')
        
        # Make predictions
        climate_risk_score = models['regression'].predict(regression_data)[0]
        flood_risk_prediction = models['classification'].predict(classification_data)[0]
        flood_risk_proba = models['classification'].predict_proba(classification_data)[0]
        
        # Convert predictions to user-friendly format
        risk_categories = ['Low', 'Medium', 'High']
        flood_risk_category = risk_categories[flood_risk_prediction] if flood_risk_prediction < len(risk_categories) else 'Unknown'
        
        # Create probability distribution
        risk_probabilities = {
            risk_categories[i]: float(flood_risk_proba[i]) if i < len(flood_risk_proba) else 0.0
            for i in range(len(risk_categories))
        }
        
        results = {
            'success': True,
            'climate_risk_score': float(climate_risk_score),
            'flood_risk_category': flood_risk_category,
            'risk_probabilities': risk_probabilities,
            'risk_level': 'Low' if climate_risk_score < 0.33 else 'Medium' if climate_risk_score < 0.67 else 'High'
        }
        
        print(f"Prediction successful: {results['risk_level']}")
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    try:
        # Check what's actually loaded
        models_status = {
            'loaded': len(models) > 0,
            'available': list(models.keys()),
            'count': len(models)
        }
        
        encoders_status = {
            'loaded': len(encoders) > 0,
            'available': list(encoders.keys()),
            'count': len(encoders)
        }
        
        scalers_status = {
            'loaded': len(scalers) > 0,
            'available': list(scalers.keys()),
            'count': len(scalers)
        }
        
        feature_names_status = {
            'loaded': len(feature_names) > 0,
            'available': list(feature_names.keys()),
            'count': len(feature_names)
        }
        
        # Check file existence
        model_files = [
            'best_regression_model_linear.pkl',
            'best_classification_model_logistic.pkl',
            'regression_scaler.pkl',
            'classification_scaler.pkl',
            'regression_encoders.pkl',
            'classification_encoders.pkl',
            'feature_names.json'
        ]
        
        file_status = {file: os.path.exists(file) for file in model_files}
        
        overall_status = 'healthy' if all([
            len(models) > 0,
            len(encoders) > 0,
            len(scalers) > 0,
            len(feature_names) > 0
        ]) else 'unhealthy'
        
        return jsonify({
            'status': overall_status,
            'platform': 'Render.com',
            'models': models_status,
            'encoders': encoders_status,
            'scalers': scalers_status,
            'feature_names': feature_names_status,
            'files': file_status,
            'debug': {
                'cwd': os.getcwd(),
                'all_files': os.listdir('.') if os.path.exists('.') else []
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'platform': 'Render.com'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Eco-Tourism Climate Risk Prediction API...")
    print("Platform: Render.com")
    print("=" * 50)
    
    # Try to load models
    models_loaded = load_models_and_processors()
    if not models_loaded:
        print("❌ WARNING: Models not loaded. Some functionality may not work.")
        print("Please ensure all .pkl model files are present in the repository.")
    else:
        print("✅ All models loaded successfully!")
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 10000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Server starting on port {port}")
    print(f"Debug mode: {debug_mode}")
    print("=" * 50)
    
    # For production deployment on Render
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
