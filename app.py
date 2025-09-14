"""
Flask Backend API for Eco-Tourism Climate Risk Prediction
Optimized for Render.com deployment
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
    """Load all trained models and data processors"""
    global models, scalers, encoders, feature_names
    
    try:
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
            print("Please run the main training script first to generate model files.")
            return False
        
        # Load models
        models['regression'] = joblib.load('best_regression_model_linear.pkl')
        models['classification'] = joblib.load('best_classification_model_logistic.pkl')
        
        # Load scalers
        scalers['regression'] = joblib.load('regression_scaler.pkl')
        scalers['classification'] = joblib.load('classification_scaler.pkl')
        
        # Load encoders
        encoders['regression'] = joblib.load('regression_encoders.pkl')
        encoders['classification'] = joblib.load('classification_encoders.pkl')
        
        # Load feature names
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        print("All models and processors loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

def preprocess_input_data(data, task_type='regression'):
    """Preprocess input data same way as training data"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Handle categorical encoding
        categorical_features = ['Vegetation_Type', 'Soil_Type', 'Country']
        for feature in categorical_features:
            if feature in df.columns and feature in encoders[task_type]:
                encoder = encoders[task_type][feature]
                # Handle unknown categories by assigning 0
                df[feature] = df[feature].apply(
                    lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else 0
                )
        
        # Handle boolean column
        if 'Protected_Area_Status' in df.columns:
            df['Protected_Area_Status'] = df['Protected_Area_Status'].astype(int)
        
        # Select only feature columns needed for the model
        feature_cols = feature_names[f'{task_type}_features']
        df_features = df[feature_cols]
        
        # Scale features
        scaled_features = scalers[task_type].transform(df_features)
        
        return scaled_features
        
    except Exception as e:
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
    """API endpoint for predictions"""
    try:
        # Get input data from request
        input_data = request.json
        
        # Validate required fields (matching the actual dataset structure)
        required_fields = [
            'Latitude', 'Longitude', 'Vegetation_Type', 'Biodiversity_Index',
            'Protected_Area_Status', 'Elevation_m', 'Slope_Degree', 'Soil_Type',
            'Air_Quality_Index', 'Average_Temperature_C', 'Tourist_Attractions',
            'Accessibility_Score', 'Tourist_Capacity_Limit'
        ]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Add default values for fields that might be missing but are needed by the model
        if 'Country' not in input_data:
            input_data['Country'] = 'USA'  # Default country
        if 'Flood_Risk_Index' not in input_data:
            input_data['Flood_Risk_Index'] = 0.3  # Default moderate flood risk
        if 'Drought_Risk_Index' not in input_data:
            input_data['Drought_Risk_Index'] = 0.3  # Default moderate drought risk
        if 'Temperature_C' not in input_data:
            input_data['Temperature_C'] = input_data.get('Average_Temperature_C', 20.0)
        if 'Annual_Rainfall_mm' not in input_data:
            input_data['Annual_Rainfall_mm'] = 1000.0  # Default rainfall
        if 'Soil_Erosion_Risk' not in input_data:
            input_data['Soil_Erosion_Risk'] = 0.2  # Default low erosion risk
        if 'Current_Tourist_Count' not in input_data:
            input_data['Current_Tourist_Count'] = input_data.get('Tourist_Capacity_Limit', 500) * 0.6
        if 'Human_Activity_Index' not in input_data:
            input_data['Human_Activity_Index'] = 0.4  # Default moderate activity
        if 'Conservation_Investment_USD' not in input_data:
            input_data['Conservation_Investment_USD'] = 100000.0  # Default investment
        if 'Climate_Risk_Score' not in input_data:
            input_data['Climate_Risk_Score'] = 0.4  # Default moderate climate risk
        
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
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'available_models': list(models.keys()),
        'platform': 'Render.com'
    })

# Error handlers for better user experience
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Eco-Tourism Climate Risk Prediction API...")
    print("Platform: Render.com")
    
    # Try to load models
    models_loaded = load_models_and_processors()
    if not models_loaded:
        print("Warning: Models not loaded. Some functionality may not work.")
        print("Please ensure all .pkl model files are present in the repository.")
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 10000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Server starting on port {port}")
    print(f"Debug mode: {debug_mode}")
    
    # For production deployment on Render
    app.run(debug=debug_mode, host='0.0.0.0', port=port)