"""
Flask Backend API for Eco-Tourism Climate Risk Prediction
Optimized for Render.com deployment with models/ directory support
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Directory where all model and processor files live
MODEL_DIR = 'models'

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global storage for loaded models and processors
models = {}
scalers = {}
encoders = {}
feature_names = {}

def load_models_and_processors():
    """Load all trained models and data processors from models/ directory"""
    global models, scalers, encoders, feature_names

    try:
        print("Starting model loading process...")
        print(f"CWD: {os.getcwd()}, files: {os.listdir('.')}")
        # List expected files under models/
        expected = [
            'best_regression_model_linear.pkl',
            'best_classification_model_logistic.pkl',
            'regression_scaler.pkl',
            'classification_scaler.pkl',
            'regression_encoders.pkl',
            'classification_encoders.pkl'
        ]
        missing = [f for f in expected if not os.path.exists(os.path.join(MODEL_DIR, f))]
        if missing:
            print(f"Missing model files in {MODEL_DIR}/: {missing}")
            return False

        # Load regression and classification models
        models['regression'] = joblib.load(os.path.join(MODEL_DIR, 'best_regression_model_linear.pkl'))
        models['classification'] = joblib.load(os.path.join(MODEL_DIR, 'best_classification_model_logistic.pkl'))

        # Load scalers
        scalers['regression'] = joblib.load(os.path.join(MODEL_DIR, 'regression_scaler.pkl'))
        scalers['classification'] = joblib.load(os.path.join(MODEL_DIR, 'classification_scaler.pkl'))

        # Load encoders
        encoders['regression'] = joblib.load(os.path.join(MODEL_DIR, 'regression_encoders.pkl'))
        encoders['classification'] = joblib.load(os.path.join(MODEL_DIR, 'classification_encoders.pkl'))

        # Load feature names (JSON in project root)
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)

        print("✅ All models and processors loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading models/processors: {e}")
        import traceback; traceback.print_exc()
        return False

def preprocess_input_data(data, task_type='regression'):
    """Preprocess input data same way as training"""
    try:
        # Validate that loaders ran
        if not models:
            raise Exception("Models not loaded")

        if task_type not in encoders or task_type not in scalers or f"{task_type}_features" not in feature_names:
            raise Exception(f"Missing encoders/scalers/features for task '{task_type}'")

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical features
        for feat in ['Vegetation_Type', 'Soil_Type', 'Country']:
            if feat in df.columns and feat in encoders[task_type]:
                encoder = encoders[task_type][feat]
                df[feat] = df[feat].apply(
                    lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else 0
                )

        # Boolean to int
        if 'Protected_Area_Status' in df.columns:
            df['Protected_Area_Status'] = df['Protected_Area_Status'].astype(int)

        # Select and scale features
        cols = feature_names[f"{task_type}_features"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise Exception(f"Missing input features: {missing}")
        features = df[cols]
        return scalers[task_type].transform(features)

    except Exception as e:
        raise Exception(f"Error preprocessing data: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not models:
            return jsonify({'error': 'Models not loaded', 'success': False}), 500

        input_data = request.json or {}
        # Required inputs
        required = [
            'Latitude','Longitude','Vegetation_Type','Biodiversity_Index',
            'Protected_Area_Status','Elevation_m','Slope_Degree','Soil_Type',
            'Air_Quality_Index','Average_Temperature_C','Tourist_Attractions',
            'Accessibility_Score','Tourist_Capacity_Limit'
        ]
        missing = [f for f in required if f not in input_data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}', 'success': False}), 400

        # Defaults
        defaults = {
            'Country':'USA','Flood_Risk_Index':0.3,'Drought_Risk_Index':0.3,
            'Temperature_C': input_data.get('Average_Temperature_C',20.0),
            'Annual_Rainfall_mm':1000.0,'Soil_Erosion_Risk':0.2,
            'Current_Tourist_Count': input_data['Tourist_Capacity_Limit']*0.6,
            'Human_Activity_Index':0.4,'Conservation_Investment_USD':100000.0,
            'Climate_Risk_Score':0.4
        }
        for k,v in defaults.items():
            input_data.setdefault(k,v)

        reg_X = preprocess_input_data(input_data, 'regression')
        cls_X = preprocess_input_data(input_data, 'classification')

        score = models['regression'].predict(reg_X)[0]
        cls = models['classification'].predict(cls_X)[0]
        proba = models['classification'].predict_proba(cls_X)[0]

        cats = ['Low','Medium','High']
        flood_cat = cats[cls] if cls < len(cats) else 'Unknown'
        probs = {cats[i]: float(proba[i]) for i in range(len(cats))}

        result = {
            'success': True,
            'climate_risk_score': float(score),
            'flood_risk_category': flood_cat,
            'risk_probabilities': probs,
            'risk_level': 'Low' if score<0.33 else 'Medium' if score<0.67 else 'High'
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    status = 'healthy' if models and encoders and scalers and feature_names else 'unhealthy'
    return jsonify({
        'status': status,
        'models_loaded': bool(models),
        'available_models': list(models.keys()),
        'encoders_loaded': bool(encoders),
        'available_encoders': list(encoders.keys()),
        'scalers_loaded': bool(scalers),
        'available_scalers': list(scalers.keys()),
        'feature_names_loaded': bool(feature_names),
        'available_feature_sets': list(feature_names.keys())
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error':'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error':'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting API on Render.com...")
    loaded = load_models_and_processors()
    if not loaded:
        print("WARNING: Models/processors failed to load.")
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('FLASK_ENV')=='development'
    app.run(host='0.0.0.0', port=port, debug=debug)
