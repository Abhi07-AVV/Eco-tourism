// DOM Elements
const form = document.getElementById('predictionForm');
const resultsSection = document.getElementById('resultsSection');
const loadingSection = document.getElementById('loadingSection');
const errorSection = document.getElementById('errorSection');
const predictBtn = document.getElementById('predictBtn');

// Range input elements
const biodiversityRange = document.getElementById('biodiversity_index');
const biodiversityValue = document.getElementById('biodiversity_value');
const slopeRange = document.getElementById('slope_degree');
const slopeValue = document.getElementById('slope_value');
const accessibilityRange = document.getElementById('accessibility');
const accessibilityValue = document.getElementById('accessibility_value');

// Initialize range value displays
function initializeRangeInputs() {
    // Biodiversity Index
    biodiversityRange.addEventListener('input', function() {
        biodiversityValue.textContent = parseFloat(this.value).toFixed(2);
    });

    // Slope Degree
    slopeRange.addEventListener('input', function() {
        slopeValue.textContent = this.value + '°';
    });

    // Accessibility Score
    accessibilityRange.addEventListener('input', function() {
        accessibilityValue.textContent = this.value;
    });
}

// Show/Hide sections
function showSection(section) {
    section.style.display = 'block';
    section.classList.add('fade-in');
}

function hideSection(section) {
    section.style.display = 'none';
    section.classList.remove('fade-in');
}

function hideAllResults() {
    hideSection(resultsSection);
    hideSection(loadingSection);
    hideSection(errorSection);
}

// Form validation
function validateForm(formData) {
    const requiredFields = [
        'Latitude', 'Longitude', 'Vegetation_Type', 'Biodiversity_Index',
        'Protected_Area_Status', 'Elevation_m', 'Slope_Degree', 'Soil_Type',
        'Air_Quality_Index', 'Average_Temperature_C', 'Tourist_Attractions',
        'Accessibility_Score', 'Tourist_Capacity_Limit'
    ];

    const errors = [];

    requiredFields.forEach(field => {
        if (!formData[field] && formData[field] !== 0 && formData[field] !== false) {
            errors.push(`${field.replace(/_/g, ' ')} is required`);
        }
    });

    // Additional validation
    if (formData.Latitude < -90 || formData.Latitude > 90) {
        errors.push('Latitude must be between -90 and 90');
    }

    if (formData.Longitude < -180 || formData.Longitude > 180) {
        errors.push('Longitude must be between -180 and 180');
    }

    if (formData.Biodiversity_Index < 0 || formData.Biodiversity_Index > 1) {
        errors.push('Biodiversity Index must be between 0 and 1');
    }

    if (formData.Air_Quality_Index < 0 || formData.Air_Quality_Index > 500) {
        errors.push('Air Quality Index must be between 0 and 500');
    }

    return errors;
}

// Collect form data
function collectFormData() {
    const formData = new FormData(form);
    const data = {};

    // Convert form data to object
    for (let [key, value] of formData.entries()) {
        if (key === 'Protected_Area_Status') {
            data[key] = value === 'true';
        } else if (['Latitude', 'Longitude', 'Biodiversity_Index', 'Elevation_m', 
                   'Slope_Degree', 'Air_Quality_Index', 'Average_Temperature_C', 
                   'Tourist_Attractions', 'Accessibility_Score', 'Tourist_Capacity_Limit'].includes(key)) {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    }

    return data;
}

// Display results
function displayResults(results) {
    // Climate Risk Score
    const climateScore = document.getElementById('climateScore');
    const climateFill = document.getElementById('climateFill');
    const climateRiskLevel = document.getElementById('climateRiskLevel');

    climateScore.textContent = results.climate_risk_score.toFixed(3);
    
    // Set gauge fill and color based on risk level
    const scorePercentage = results.climate_risk_score * 100;
    climateFill.style.width = scorePercentage + '%';
    
    let fillColor, riskClass, riskText;
    if (results.climate_risk_score < 0.33) {
        fillColor = 'linear-gradient(90deg, #27ae60, #2ecc71)';
        riskClass = 'low';
        riskText = 'Low Risk';
    } else if (results.climate_risk_score < 0.67) {
        fillColor = 'linear-gradient(90deg, #f39c12, #e67e22)';
        riskClass = 'medium';
        riskText = 'Medium Risk';
    } else {
        fillColor = 'linear-gradient(90deg, #e74c3c, #c0392b)';
        riskClass = 'high';
        riskText = 'High Risk';
    }
    
    climateFill.style.background = fillColor;
    climateRiskLevel.textContent = riskText;
    climateRiskLevel.className = 'risk-level ' + riskClass;

    // Flood Risk Category
    const floodCategory = document.getElementById('floodCategory');
    const riskLower = results.flood_risk_category.toLowerCase();
    floodCategory.textContent = results.flood_risk_category;
    floodCategory.className = 'flood-category ' + riskLower;

    // Probability Bars
    const probabilityBars = document.getElementById('probabilityBars');
    probabilityBars.innerHTML = '';

    Object.entries(results.risk_probabilities).forEach(([category, probability]) => {
        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar';

        const label = document.createElement('div');
        label.className = 'probability-label';
        label.textContent = category;

        const fillContainer = document.createElement('div');
        fillContainer.className = 'probability-fill';

        const fillBar = document.createElement('div');
        fillBar.className = 'probability-fill-bar ' + category.toLowerCase();
        fillBar.style.width = (probability * 100) + '%';

        const value = document.createElement('div');
        value.className = 'probability-value';
        value.textContent = (probability * 100).toFixed(1) + '%';

        fillContainer.appendChild(fillBar);
        barContainer.appendChild(label);
        barContainer.appendChild(fillContainer);
        barContainer.appendChild(value);
        probabilityBars.appendChild(barContainer);
    });

    showSection(resultsSection);
}

// Display error
function displayError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    showSection(errorSection);
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    hideAllResults();
    
    // Collect and validate form data
    const formData = collectFormData();
    const validationErrors = validateForm(formData);
    
    if (validationErrors.length > 0) {
        displayError('Please fix the following errors:\n' + validationErrors.join('\n'));
        return;
    }

    // Show loading
    showSection(loadingSection);
    predictBtn.disabled = true;

    try {
        // Make API request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (result.success) {
            hideSection(loadingSection);
            displayResults(result);
        } else {
            hideSection(loadingSection);
            displayError(result.error || 'An error occurred during prediction');
        }
    } catch (error) {
        hideSection(loadingSection);
        displayError('Network error: Unable to connect to the prediction service. Please check if the server is running.');
        console.error('Error:', error);
    } finally {
        predictBtn.disabled = false;
    }
}

// Sample data function for testing
function fillSampleData() {
    document.getElementById('latitude').value = '25.7617';
    document.getElementById('longitude').value = '-80.1918';
    document.getElementById('vegetation_type').value = 'Wetland';
    document.getElementById('biodiversity_index').value = '0.75';
    document.getElementById('biodiversity_value').textContent = '0.75';
    document.querySelector('input[name="Protected_Area_Status"][value="true"]').checked = true;
    document.getElementById('elevation').value = '2';
    document.getElementById('slope_degree').value = '5';
    document.getElementById('slope_value').textContent = '5°';
    document.getElementById('soil_type').value = 'Sandy';
    document.getElementById('air_quality').value = '85';
    document.getElementById('temperature').value = '24.5';
    document.getElementById('tourist_attractions').value = '6';
    document.getElementById('accessibility').value = '8';
    document.getElementById('accessibility_value').textContent = '8';
    document.getElementById('tourist_capacity').value = '500';
}

// Add sample data button (for testing)
function addSampleDataButton() {
    const submitSection = document.querySelector('.submit-section');
    const sampleBtn = document.createElement('button');
    sampleBtn.type = 'button';
    sampleBtn.innerHTML = '<i class="fas fa-flask"></i> Fill Sample Data';
    sampleBtn.style.marginRight = '20px';
    sampleBtn.style.background = 'linear-gradient(135deg, #27ae60, #2ecc71)';
    sampleBtn.style.color = 'white';
    sampleBtn.style.border = 'none';
    sampleBtn.style.padding = '15px 30px';
    sampleBtn.style.fontSize = '1rem';
    sampleBtn.style.fontWeight = '600';
    sampleBtn.style.borderRadius = '50px';
    sampleBtn.style.cursor = 'pointer';
    sampleBtn.style.transition = 'all 0.3s ease';
    sampleBtn.style.boxShadow = '0 4px 15px rgba(39, 174, 96, 0.3)';
    
    sampleBtn.addEventListener('click', fillSampleData);
    sampleBtn.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-2px)';
        this.style.boxShadow = '0 6px 20px rgba(39, 174, 96, 0.4)';
    });
    sampleBtn.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
        this.style.boxShadow = '0 4px 15px rgba(39, 174, 96, 0.3)';
    });
    
    submitSection.insertBefore(sampleBtn, predictBtn);
}

// Initialize the application
function initializeApp() {
    initializeRangeInputs();
    form.addEventListener('submit', handleFormSubmit);
    addSampleDataButton();
    
    // Check if API is available
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            if (!data.models_loaded) {
                console.warn('Models not loaded on server. Some functionality may not work.');
            }
        })
        .catch(error => {
            console.warn('Could not check API health:', error);
        });
}

// Start the application when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
