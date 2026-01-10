# AutoML Application

A comprehensive, production-ready **Automated Machine Learning (AutoML)** platform with an interactive UI, advanced ensemble methods, model registry, hyperparameter tuning, and real-time monitoring. Built with FastAPI backend and vanilla JavaScript frontend.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Features

### Core ML Capabilities
- **Multi-Algorithm Support**: RandomForest, XGBoost, LightGBM, Linear Regression, SVM, KNeighbors
- **Auto/Manual Training Modes**: Automatic model selection or manual multi-model selection
- **Ensemble Methods**: Stacking and Voting ensemble techniques with customizable configuration
- **Hyperparameter Tuning**: Advanced Bayesian optimization and grid search tuning
- **Problem Detection**: Automatic regression/classification/time-series detection

### Advanced Features
- **Model Registry & Versioning**: Track and manage multiple model versions with metadata
- **Model Explainability**: SHAP-based feature importance and model interpretation
- **Data Validation**: Schema validation, outlier detection, and data quality checks
- **Drift Monitoring**: Real-time model performance drift detection and alerts
- **Time Series Support**: Automatic time-series preprocessing and forecasting
- **Batch Deployment**: Deploy multiple models in production with scheduling

### User Interface
- **Interactive Training Studio**: 
  - Real-time training visualization with animated model flows
  - Multi-model selection with card-based UI
  - Ensemble method selector (Stacking/Voting)
  - Live training console with per-model logging
  - Progress tracking and performance metrics
  
- **Dashboard**: Overview of registered models, training history, and system metrics
- **Prediction Hub**: Make predictions using registered models with time-series forecast support
- **Model Monitor**: Track model drift and performance degradation
- **User Records**: View training history and model performance metrics
- **Authentication**: Secure login/registration system

### Technical Highlights
- **Asynchronous Training**: Non-blocking background training with real-time status updates
- **Model Persistence**: Joblib-based model serialization with metadata tracking
- **Database**: SQLAlchemy ORM with SQLite backend for job and user management
- **REST API**: Comprehensive RESTful endpoints for all operations
- **Real-time Updates**: WebSocket-ready polling mechanism for live progress tracking

---

## 📋 Tech Stack

### Backend
- **Framework**: FastAPI (async Python web framework)
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, SHAP
- **Data Processing**: pandas, NumPy, Scikit-learn
- **Optimization**: Optuna (Bayesian hyperparameter tuning)
- **Database**: SQLAlchemy ORM with SQLite
- **Serialization**: joblib, JSON

### Frontend
- **HTML/CSS/JavaScript**: Vanilla JavaScript (no frameworks)
- **Charting**: Chart.js for interactive visualizations
- **Styling**: Custom CSS with professional theme and animations
- **Visualization**: SVG-based animated training flow diagrams

### Deployment
- **Server**: Uvicorn (ASGI application server)
- **Container**: Docker-ready structure
- **Package Manager**: pip

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/automl_application.git
   cd automl_application
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   ```bash
   python fix_database.py
   ```

5. **Start the application**
   ```bash
   python main.py
   ```
   The application will start at `http://localhost:8081`

---

## 🚀 Quick Start

### Using the Web UI

1. **Open your browser** and navigate to `http://localhost:8081`
2. **Register an account** or login with credentials
3. **Go to Training Studio**:
   - Upload a CSV file
   - Select features and target variable
   - Choose training mode:
     - **Auto Mode**: System automatically selects best models
     - **Manual Mode**: Manually select multiple models to train
   - Select ensemble method (Stacking/Voting) if enabling ensembles
   - Click "Start Training"
4. **Monitor Progress**: Watch real-time training visualization with:
   - Model execution flow with animated connectors
   - Per-model progress bars
   - Live console logs
   - Model detail modal for individual model metrics
5. **View Results**: Check leaderboard, validation metrics, and registered models
6. **Make Predictions**: Go to Prediction Hub and select a registered model

### Using the REST API

#### Train a Model
```bash
curl -X POST http://localhost:8081/train \
  -F "file=@data.csv" \
  -F "target=target_column" \
  -F "enable_ensemble=true" \
  -F "ensemble_method=stacking" \
  -F "selected_models=[\"RandomForest\",\"XGBoost\"]"
```

#### Get Training Status
```bash
curl http://localhost:8081/status
```

#### List Registered Models
```bash
curl http://localhost:8081/models
```

#### Make Predictions
```bash
curl -X POST http://localhost:8081/predict \
  -H "Content-Type: application/json" \
  -d '{"version_id":"v_XXXXX","data":[[1,2,3,4]]}'
```

#### Get Model Forecast (Time Series)
```bash
curl http://localhost:8081/forecast/v_XXXXX/12
```

---

## 📁 Project Structure

```
automl_application/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── schema.json                      # Data schema definition
├── static/                          # Frontend assets
│   ├── index.html                  # Main SPA HTML
│   ├── script.js                   # Frontend JavaScript logic
│   └── style.css                   # Styling and animations
├── app_deployment/                  # Deployment utilities
│   └── app.py                       # Batch deployment app
├── classification/                  # Classification preprocessing
│   └── Classification_preprocessing.py
├── regression/                      # Regression preprocessing
│   └── regression_preprocessing.py
├── data_validation/                 # Data validation module
│   ├── __init__.py
│   └── validator.py
├── ingestion_of_data/              # Data loading module
│   └── data_loader.py
├── model_registry/                  # Trained model versions
│   └── v_XXXXX/
│       ├── metadata.json           # Model metadata
│       └── context.csv             # Training context
├── model_registry_system/          # Model registry management
│   ├── model_registry.py
│   └── batch_deployment.py
├── model_zoo/                       # Model definitions
│   └── models.py
├── ensemble/                        # Ensemble methods
│   ├── __init__.py
│   └── ensemble_methods.py
├── hyper_parameter_tuning/         # Hyperparameter optimization
│   ├── hyper_parameter.py
│   └── advanced_tuning.py
├── model_explainability/            # Model interpretation
│   ├── __init__.py
│   └── explainer.py
├── monitoring/                      # Model monitoring
│   ├── __init__.py
│   └── model_monitor.py
├── drifting_of_model/              # Drift detection
│   └── drifting.py
├── problem_detector/                # Problem type detection
│   └── problem_detector.py
├── Timeseries_auto_module/         # Time series utilities
│   └── timeseriespreprocessing.py
├── anomaly/                         # Anomaly detection
│   └── unsupervised.py
├── sample_data/                     # Sample datasets for testing
│   ├── creditcard.csv
│   ├── fertility.csv
│   └── Student_Performance.csv
└── monitoring_logs/                 # Training logs directory
```

---

## 🎯 Usage Examples

### Example 1: Basic Classification Training
```python
# Using the UI:
1. Upload a classification dataset (e.g., iris.csv)
2. Select features and target variable
3. Set to "Manual Mode"
4. Select models: RandomForest, XGBoost
5. Enable Ensemble with Stacking method
6. Click "Start Training"
```

### Example 2: Time Series Forecasting
```python
# Using the UI:
1. Upload a time series dataset with datetime column
2. Select datetime column as temporal reference
3. Select the value column as target
4. Train models (system auto-detects time series)
5. Go to Prediction Hub
6. Select the trained model and set forecast horizon (e.g., 12 months)
7. View forecasted values
```

### Example 3: Auto Ensemble with Top Models
```python
# Using the UI:
1. Go to Training Studio (Auto Mode)
2. Upload dataset
3. Configure target and preprocessing
4. Enable Ensemble with Voting method
5. Start training - system automatically:
   - Trains all available algorithms
   - Selects top-performing models
   - Creates ensemble from top models
```

### Example 4: Model Monitoring
```python
# Using the UI:
1. Go to Drifting Monitor
2. Select a trained model
3. View real-time metrics:
   - Current model performance
   - Performance over time
   - Drift indicators
   - Alert thresholds
```

---

## 🔌 API Endpoints

### Authentication
- `POST /login` - User login
- `POST /register` - Register new user

### Training
- `POST /train` - Start a new training job
- `GET /status` - Get current training status and logs
- `GET /leaderboard` - Get model leaderboard

### Models
- `GET /models` - List all registered models
- `GET /models/{version_id}` - Get model details
- `POST /predict` - Make predictions using a model
- `GET /forecast/{version_id}/{horizon}` - Forecast time series
- `GET /download/{version_id}` - Download model file

### Monitoring
- `GET /monitor/{version_id}` - Get model monitoring data
- `GET /drift/{version_id}` - Get drift detection metrics

### System
- `GET /history` - Get user training history
- `GET /leaderboard` - Get leaderboard of all trained models
- `GET /explainability/{version_id}` - Get model feature importance

---

## ⚙️ Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=sqlite:///./test.db

# Server
HOST=0.0.0.0
PORT=8081
RELOAD=true

# Logging
LOG_LEVEL=INFO
```

### Training Configuration
Edit `main.py` to configure:
- Test/validation split ratio (default: 0.2)
- Number of tuning iterations (default: 10)
- Ensemble methods available (Stacking, Voting)
- Model registry path (default: `./model_registry/`)

### UI Themes
Edit `static/style.css` to customize:
- Color palette
- Animation speeds
- Card and button styles
- Modal appearance

---

## 🖼️ Key UI Features

### Training Studio
- **Multi-Model Selection**: Card-based UI for selecting multiple algorithms
- **Ensemble Configuration**: Toggle between Auto/Manual mode, select ensemble method
- **Real-Time Visualization**: 
  - Animated model flow diagram showing training progression
  - Progress bars per model
  - Live console with color-coded logs
  - Model detail modal for individual metrics
- **Training Console**: Streaming logs with filtering by model

### Dashboard
- **Model Overview**: Quick stats on registered models
- **Performance Metrics**: Charts showing model scores and validation results
- **Training History**: Timeline of all training jobs
- **Quick Actions**: Access frequently used operations

### Prediction Hub
- **Model Selection**: Dropdown with all registered model versions
- **Batch Prediction**: Upload CSV for batch predictions
- **Time Series Forecasting**: Built-in horizon selector for forecasting
- **Results Export**: Download predictions as CSV

---

## 🔍 Advanced Features

### Hyperparameter Tuning
The system uses Bayesian optimization (Optuna) for intelligent hyperparameter search:
```
- Efficient parameter space exploration
- Adaptive sampling based on previous trials
- Early stopping for poor performers
- Parallel trial execution support
```

### Ensemble Methods

**Stacking**:
- Trains base models on training data
- Uses base model predictions as features for meta-learner
- Optimal for combining diverse models

**Voting**:
- Hard voting: Majority class prediction
- Soft voting: Average probability predictions
- Fast and interpretable ensemble

### Data Validation
- Type checking and schema validation
- Outlier detection
- Missing value handling
- Feature scaling consistency
- Train/test distribution alignment

### Model Explainability
- SHAP values for feature importance
- Partial dependence plots
- Model decision analysis
- Feature contribution rankings

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t automl-app .
```

### Run Container
```bash
docker run -p 8081:8081 -v $(pwd)/model_registry:/app/model_registry automl-app
```

---

## 📊 Sample Results

After training on the sample datasets, you can expect:

**Credit Card Fraud Detection**:
- XGBoost: ~99.5% ROC-AUC
- RandomForest: ~98.9% ROC-AUC
- Stacking Ensemble: ~99.7% ROC-AUC

**Student Performance Prediction**:
- Linear Regression: ~0.85 R²
- RandomForest: ~0.92 R²
- XGBoost: ~0.93 R²

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Windows - Find and kill process on port 8081
netstat -ano | findstr :8081
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8081 | xargs kill -9
```

### Database Errors
```bash
# Reset database
python fix_database.py
```

### Missing Dependencies
```bash
# Reinstall all requirements
pip install --upgrade -r requirements.txt
```

### Training Hangs
- Check server logs for errors
- Verify dataset is not too large (>1GB recommended max)
- Ensure sufficient system memory (8GB+ recommended)

---

## 📈 Performance Tips

1. **Dataset Size**: Optimal range 1MB - 500MB for quick iterations
2. **Feature Count**: 5-50 features recommended for auto-tuning
3. **Training Time**: 2-30 minutes depending on data size and model count
4. **Memory**: 4GB minimum, 8GB+ recommended for larger datasets
5. **Ensemble Selection**: Use 2-5 models for best stacking performance

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
python -m pytest

# Format code
black .

# Lint
flake8 .
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML algorithms from [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/)
- Hyperparameter tuning with [Optuna](https://optuna.org/)
- Model explainability with [SHAP](https://shap.readthedocs.io/)
- Visualization with [Chart.js](https://www.chartjs.org/)

---

## 📧 Contact & Support

For issues, feature requests, or questions:
- Open an GitHub Issue
- Contact the maintainers
- Check the [Discussion board](https://github.com/yourusername/automl_application/discussions)

---

## 🎓 Learning Resources

- [AutoML Concepts](https://en.wikipedia.org/wiki/Automated_machine_learning)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Tutorials](https://xgboost.readthedocs.io/en/latest/tutorials/)
- [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Production Ready ✅
