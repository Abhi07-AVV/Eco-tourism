# Eco-Tourism Climate Risk Prediction

A machine learning-powered web application that predicts climate risks for eco-tourism destinations. The application uses both regression and classification models to provide comprehensive risk assessments for eco-tourism sites.

## 🚀 Live Demo
- **Production URL**: [Your Render.com URL will be here after deployment]
- **API Health Check**: `/api/health`

## ✨ Features

- **Climate Risk Score Prediction**: Uses regression model to predict numerical risk scores (0-1)
- **Flood Risk Category Prediction**: Uses classification model for Low/Medium/High risk categories
- **Interactive Web Interface**: User-friendly form for data input with validation
- **Real-time Predictions**: RESTful API endpoints for instant predictions
- **Risk Probability Distribution**: Visual representation of prediction confidence
- **Health Monitoring**: Built-in health check endpoint for system monitoring
- **Mobile Responsive**: Works seamlessly across all device sizes

## 🛠️ Tech Stack

- **Backend**: Python 3.9, Flask, Gunicorn
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Deployment**: Render.com
- **Model Serialization**: joblib
- **API**: RESTful endpoints with JSON responses

## 📊 Model Information

The application uses pre-trained machine learning models:
- **Regression Model**: Linear regression for climate risk score prediction
- **Classification Model**: Logistic regression for flood risk categorization
- **Feature Engineering**: StandardScaler for normalization, LabelEncoder for categorical variables
- **Prediction Features**: 20+ environmental, geographical, and tourism-related features

## 🏗️ Project Structure

```
eco-tourism-climate-risk/
├── app.py                              # Main Flask application
├── requirements.txt                    # Python dependencies
├── runtime.txt                        # Python version specification
├── feature_names.json                 # Model feature specifications
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore rules
├── static/                            # Frontend static files
│   ├── styles.css                     # Application styling
│   └── script.js                      # Frontend JavaScript
├── templates/                         # HTML templates
│   └── index.html                     # Main application interface
└── models/                            # Machine learning models
    ├── best_regression_model_linear.pkl
    ├── best_classification_model_logistic.pkl
    ├── regression_scaler.pkl
    ├── classification_scaler.pkl
    ├── regression_encoders.pkl
    └── classification_encoders.pkl
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/eco-tourism-climate-risk.git
cd eco-tourism-climate-risk
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the application**:
```bash
python app.py
```

5. **Access the application**:
   - Open http://localhost:10000 in your browser
   - API documentation available at `/api/health`

### Deploy to Render.com

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com)
   - Create new Web Service from GitHub repository
   - Use default settings (auto-detected Python/Flask)
   - Deploy automatically

## 📡 API Endpoints

### Health Check
```http
GET /api/health
```

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["regression", "classification"],
  "platform": "Render.com"
}
```

### Risk Prediction
```http
POST /api/predict
Content-Type: application/json
```

**Request Body**:
```json
{
  "Latitude": 25.7617,
  "Longitude": -80.1918,
  "Vegetation_Type": "Wetland",
  "Biodiversity_Index": 0.75,
  "Protected_Area_Status": true,
  "Elevation_m": 2,
  "Slope_Degree": 5,
  "Soil_Type": "Sandy",
  "Air_Quality_Index": 85,
  "Average_Temperature_C": 24.5,
  "Tourist_Attractions": 6,
  "Accessibility_Score": 0.8,
  "Tourist_Capacity_Limit": 500
}
```

**Response**:
```json
{
  "success": true,
  "climate_risk_score": 0.456,
  "flood_risk_category": "Medium",
  "risk_probabilities": {
    "Low": 0.2,
    "Medium": 0.6,
    "High": 0.2
  },
  "risk_level": "Medium"
}
```

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 10000)
- `FLASK_ENV`: Set to 'development' for debug mode

### Model Requirements
Ensure these files are present in the root directory:
- `best_regression_model_linear.pkl`
- `best_classification_model_logistic.pkl`
- `regression_scaler.pkl`
- `classification_scaler.pkl`
- `regression_encoders.pkl`
- `classification_encoders.pkl`
- `feature_names.json`

## 🧪 Testing

### Manual Testing
1. Visit the deployed application URL
2. Fill out the prediction form with sample data
3. Click "Fill Sample Data" for quick testing
4. Submit and verify predictions are returned

### API Testing
```bash
# Health check
curl https://your-app.onrender.com/api/health

# Prediction test
curl -X POST https://your-app.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Latitude": 25.7617, "Longitude": -80.1918, ...}'
```

## 📊 Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| Latitude | Float | -90 to 90 | Geographic latitude |
| Longitude | Float | -180 to 180 | Geographic longitude |
| Vegetation_Type | String | Forest/Mountain/Wetland/Grassland | Ecosystem type |
| Biodiversity_Index | Float | 0 to 1 | Biodiversity richness score |
| Protected_Area_Status | Boolean | true/false | Conservation protection status |
| Elevation_m | Integer | ≥0 | Elevation above sea level (meters) |
| Slope_Degree | Integer | 0 to 90 | Terrain slope (degrees) |
| Soil_Type | String | Sandy/Clay/Loam/Rocky | Soil composition |
| Air_Quality_Index | Integer | 0 to 500 | Air pollution level |
| Average_Temperature_C | Float | Any | Mean temperature (Celsius) |
| Tourist_Attractions | Integer | ≥0 | Number of nearby attractions |
| Accessibility_Score | Float | 1 to 10 | Transportation accessibility |
| Tourist_Capacity_Limit | Integer | ≥0 | Maximum visitor capacity |

## 🔍 Troubleshooting

### Common Issues

**Build Fails - Missing Model Files**:
```
ERROR: Missing model files: ['*.pkl']
```
- Ensure all `.pkl` files are committed to your repository
- Check file sizes aren't too large for Git (use Git LFS if needed)

**App Won't Start - Port Binding**:
```
ERROR: Failed to bind to 0.0.0.0:10000
```
- Verify `app.py` uses `host='0.0.0.0'` and `port=int(os.environ.get('PORT', 10000))`

**Prediction Errors**:
```
ERROR: Error preprocessing data
```
- Check that all required fields are provided in the request
- Verify categorical values match training data categories

### Debugging Tips
- Check Render logs in real-time during deployment
- Test the `/api/health` endpoint first
- Use browser developer tools to inspect API requests
- Verify model files load successfully in application logs

## 🚀 Performance & Scaling

### Current Limits (Free Tier)
- **Memory**: 512MB RAM
- **CPU**: Shared CPU resources
- **Storage**: 1GB SSD
- **Bandwidth**: 100GB/month
- **Sleep Policy**: Spins down after 15 minutes of inactivity

### Optimization Tips
- Models are loaded once at startup (not per request)
- Static files served efficiently by Flask
- Minimal dependencies for faster cold starts
- Error handling prevents crashes

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Flask and scikit-learn
- Deployed on Render.com
- UI inspired by modern web design principles
- Machine learning models trained on environmental and tourism data

## 📞 Support

For support, please open an issue in the GitHub repository or contact the development team.

---

**Happy Predicting! 🌍🔮**