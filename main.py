
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import shutil
import os
import io
import joblib
import json
from typing import Optional, List, Dict
from datetime import datetime
from contextlib import asynccontextmanager
import subprocess
import glob


# Database & Auth
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from passlib.context import CryptContext

# Import existing modules
try:
    from ingestion_of_data.data_loader import universal_data_loader
    from problem_detector.problem_detector import user_handled_problem
    from classification.Classification_preprocessing import Classification_preprocessing
    from regression.regression_preprocessing import Regression_preprocessing
    from Timeseries_auto_module.timeseriespreprocessing import time_series_preprocessing_final, time_series_split
    from model_zoo.models import Model_zoo, model_training_evaluation
    from model_registry_system.model_registry import model_registry, get_latest_model
    from drifting_of_model.drifting import check_data_drift, monitor_latency
    # --- NEW ADVANCED FEATURES ---
    from hyper_parameter_tuning.advanced_tuning import auto_tune_model
    from model_explainability.explainer import create_explainer, explain_predictions
    from ensemble.ensemble_methods import auto_ensemble
    from data_validation.validator import validate_data
    from monitoring.model_monitor import ModelMonitor, get_all_model_health, monitor_prediction
except ImportError as e:
    print(f"Import Error: {e}. Ensure you are running from the root directory.")

# --- DATABASE SETUP ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    filename = Column(String)
    target = Column(String)
    mode = Column(String)
    status = Column(String)
    problem_type = Column(String)  # Added to store the problem type (Classification, Regression, Time_series, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Global State
CURRENT_MODEL = None
MODEL_METADATA = None
CONTEXT_DATA = None
PIPELINE_STATUS = {
    "step": "IDLE", 
    "message": "System Ready.", 
    "logs": [],
    "active": False,
    "user": None
}

# --- PYDANTIC MODELS ---
class UserLogin(BaseModel):
    username: str
    password: str

class PredictionInput(BaseModel):
    data: List[Dict]
    version_id: Optional[str] = None

# --- HELPER FUNCTIONS ---
def update_status(step, message, user=None, msg_type="info"):
    global PIPELINE_STATUS
    PIPELINE_STATUS["step"] = step
    PIPELINE_STATUS["message"] = message
    PIPELINE_STATUS["active"] = True
    if user: PIPELINE_STATUS["user"] = user
    print(f"[{step}] {message}")
    
    # Store with type for colorful UI
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_obj = {"time": timestamp, "msg": f"[{step}] {message}", "type": msg_type}
    PIPELINE_STATUS["logs"].append(log_obj)

def load_global_model():
    global CURRENT_MODEL, MODEL_METADATA, CONTEXT_DATA
    try:
        model_path = get_latest_model()
        if model_path:
            # 1. Load Model
            model_file = os.path.join(model_path, "AutoML_Model.pkl")
            if os.path.exists(model_file):
                CURRENT_MODEL = joblib.load(model_file)
            
            # 2. Load Metadata
            meta_file = os.path.join(model_path, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    MODEL_METADATA = json.load(f)
            
            # 3. Load Context (History for TS)
            context_file = os.path.join(model_path, "context.csv")
            if os.path.exists(context_file):
                CONTEXT_DATA = pd.read_csv(context_file)
                
            update_status("DEPLOYMENT", "Model & Context loaded.")
        else:
            print("No model found in registry.")
    except Exception as e:
        print(f"Error loading model: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    load_global_model()
    yield
    # Shutdown logic (if any)

# --- APP SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="AutoML Pro", description="Enterprise AI Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explicit routes for core assets (brute force bypass for mount issues)
@app.get("/static/style.css")
def get_css():
    return FileResponse(os.path.join(STATIC_DIR, "style.css"), media_type="text/css")

@app.get("/static/script.js")
def get_js():
    return FileResponse(os.path.join(STATIC_DIR, "script.js"), media_type="application/javascript")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- ENDPOINTS ---

@app.get("/")
def home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "UI files not found. Ensure 'static' folder exists."}

@app.get("/status")
def get_status():
    return PIPELINE_STATUS

# Auth Routes
@app.post("/register")
def register(user: UserLogin, db: Session = Depends(get_db)):
    try:
        print(f"Registration attempt for user: {user.username}")
        db_user = db.query(User).filter(User.username == user.username).first()
        if db_user:
            print(f"User {user.username} already exists")
            raise HTTPException(status_code=400, detail="Username already exists")
        
        print(f"Hashing password for {user.username}...")
        hashed_pw = pwd_context.hash(user.password)
        print(f"Password hashed successfully.")
        
        new_user = User(username=user.username, hashed_password=hashed_pw)
        db.add(new_user)
        db.commit()
        print(f"User {user.username} committed to DB")
        return {"message": "User created successfully"}
    except Exception as e:
        import traceback
        print(f"Error during registration: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    print(f"Login attempt for user: {user.username}")
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user:
        print(f"Login failed: User {user.username} not found")
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    if not pwd_context.verify(user.password, db_user.hashed_password):
        print(f"Login failed: Password mismatch for user {user.username}")
        raise HTTPException(status_code=400, detail="Invalid credentials")
        
    print(f"Login successful for user: {user.username}")
    return {"message": "Login successful", "username": user.username}

@app.get("/history")
def get_history(username: str, db: Session = Depends(get_db)):
    jobs = db.query(TrainingJob).filter(TrainingJob.username == username).order_by(TrainingJob.created_at.desc()).all()
    return jobs



@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        # Padhne ke liye thoda sa data
        contents = await file.read(1024 * 1024) # 1MB peek
        df = pd.read_csv(io.BytesIO(contents))
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            columns.append({
                "name": col,
                "type": dtype,
                "suggested": "scaling" if "float" in dtype or "int" in dtype else "encoding"
            })
        return {"columns": columns, "sample": df.head(5).to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    target_column: str = Form(...),
    username: str = Form(...),
    mode: str = Form(...), # 'auto' or 'custom'
    cleaning_method: str = Form(None), 
    encoding_method: str = Form(None),
    scaling_method: str = Form(None), 
    model_preference: str = Form(None),
    column_config: str = Form(None), 
    selected_models: str = Form(None),
    ensemble_method: str = Form("stacking"),
    problem_type: str = Form("Auto"), # NEW: User can override detection
    fast_train: str = Form("true"), # Added speed flag
    enable_tuning: str = Form("false"),  # NEW: Advanced Tuning
    enable_ensemble: str = Form("false"), # NEW: Advanced Ensemble
    db: Session = Depends(get_db)
):
    # Reset Status
    global PIPELINE_STATUS
    PIPELINE_STATUS = {
        "step": "USER_DATA",
        "message": f"started by {username} in {mode.upper()} mode.",
        "logs": [],
        "active": True,
        "user": username
    }
    
    file_location = f"sample_data/{file.filename}"
    os.makedirs("sample_data", exist_ok=True)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Save Record
    new_job = TrainingJob(username=username, filename=file.filename, target=target_column, mode=mode, status="RUNNING", problem_type=problem_type)
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    job_id = new_job.id

    # Configure Pipeline Options
    try:
        col_cfg = json.loads(column_config) if (column_config and column_config != 'null') else None
    except:
        col_cfg = None

    config = {
        "mode": mode,
        "cleaning": cleaning_method,
        "encoding": encoding_method,
        "scaling": scaling_method,
        "model": model_preference,
        "job_id": job_id,
        "column_config": col_cfg,
        "problem_type": problem_type,
        "fast_train": fast_train.lower() == "true",
        "enable_tuning": enable_tuning.lower() == "true",
        "enable_ensemble": enable_ensemble.lower() == "true",
        "ensemble_method": ensemble_method or 'stacking',
        "selected_models": (json.loads(selected_models) if selected_models and selected_models != 'null' else None)
    }
    
    background_tasks.add_task(run_automl_pipeline, file_location, target_column, config)
    
    return {"message": "Training started.", "mode": mode, "job_id": job_id}

def run_automl_pipeline(file_path: str, target_column: str, config: dict):
    try:
        mode = config['mode']
        update_status("INGESTION", "Loading data...")
        df = universal_data_loader(file_path, source_type="file")
        
        if df is None:
            update_status("ERROR", "Data load failed.")
            return

        # NEW: Auto-Convert potential date columns early
        for col in df.select_dtypes(include=['object']).columns:
            if any(word in col.lower() for word in ['date', 'time', 'timestamp','Date']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if not df[col].isna().all():
                        update_status("INGESTION", f"Auto-converted '{col}' to datetime.", msg_type="info")
                except: pass
        
        # Explicitly find datetime column from config if specified
        dt_col = None
        col_cfg = config.get('column_config')
        if col_cfg:
            for col, cfg in col_cfg.items():
                if cfg.get('type') == 'datetime' or cfg.get('transform') == 'datetime':
                    dt_col = col
                    break

        # NEW: Speed Optimization - Sampling
        if config.get("fast_train") and len(df) > 10000:
            update_status("INGESTION", f"Data too large ({len(df)} rows). Speeding up via sampling...", msg_type="warning")
            sample_size = 10000
            
            # CHECK: Is it Time Series? if so, take tail (contiguous)
            is_ts = config.get('problem_type') == 'Time_series'
            if not is_ts: # Try to detect from dt_col
                is_ts = dt_col is not None

            if is_ts:
                # For Time Series, we MUST take contiguous data. Most recent is better.
                df = df.tail(sample_size)
                update_status("INGESTION", f"Using most recent {len(df)} contiguous rows for Time Series.", msg_type="success")
            elif target_column in df.columns and df[target_column].nunique() < 20: # Likely classification
                try: 
                    df = df.groupby(target_column, group_keys=False).apply(lambda x: x.sample(min(len(x), sample_size // df[target_column].nunique())))
                except:
                    df = df.sample(n=sample_size, random_state=42)
                update_status("INGESTION", f"Using optimized stratified sample of {len(df)} rows.", msg_type="success")
            else:
                df = df.sample(n=sample_size, random_state=42)
                update_status("INGESTION", f"Using optimized random sample of {len(df)} rows.", msg_type="success")


        # Problem Detection
        user_problem_type = config.get('problem_type', 'Auto')
        if user_problem_type and user_problem_type != 'Auto':
            problem_type = user_problem_type
            update_status("DETECTION", f"Using User Selected Type: {problem_type}", msg_type="warning")
        else:
            update_status("DETECTION", "Detecting problem type...")
            problem_type = user_handled_problem(df, target_column, datetime_col=dt_col)
            update_status("DETECTION", f"Detected: {problem_type}")

        # Preprocessing
        update_status("PREPROCESSING", f"Running {mode} preprocessing...")
        
        X_train, X_test, y_train, y_test = None, None, None, None
        
        # Scaling Mapping
        scaler_arg = "standard"
        if config['scaling'] == 'minmax': scaler_arg = 'minmax'
        elif config['scaling'] == 'robust': scaler_arg = 'robust'
        
        if "Classification" in problem_type or problem_type in ["Binary_classification", "Multi_classification"]:
            zoo_problem_type = "Classification"
            update_status("PREPROCESSING", f"Applying {mode} classification preprocessing...", msg_type="info")
            X_train, X_test, y_train, y_test = Classification_preprocessing(
                df, target_column, bulk=(mode == 'auto'), 
                column_config=col_cfg, bulk_scaling_method=scaler_arg
            )

        elif problem_type == "Regression":
            zoo_problem_type = "Regression"
            update_status("PREPROCESSING", f"Applying {mode} regression preprocessing...", msg_type="info")
            X_train, X_test, y_train, y_test = Regression_preprocessing(
                df, target_column, bulk=(mode == 'auto'), 
                column_config=col_cfg, bulk_scaling_method=scaler_arg
            )

        elif problem_type == "Time_series":
            zoo_problem_type = "Time_series"
            
            # If dt_col not explicitly found, try to auto-detect
            if not dt_col:
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        dt_col = col; break
                    try: 
                        pd.to_datetime(df[col], errors='raise')
                        dt_col = col; break 
                    except: continue

            if dt_col:
                update_status("PREPROCESSING", f"Time Series indexing using: {dt_col}", msg_type="info")
                processed_df = time_series_preprocessing_final(df, dt_col, target_column)
                train_df, test_df = time_series_split(processed_df)
                
                # Dropping only the original date and target. New features are kept.
                X_train = train_df.drop(columns=[target_column, dt_col])
                y_train = train_df[target_column]
                X_test = test_df.drop(columns=[target_column, dt_col])
                y_test = test_df[target_column]
            else:
                update_status("ERROR", "No datetime column found for Time Series.", msg_type="error")
                return 
        else:
            update_status("ERROR", f"Unsupported problem type: {problem_type}", msg_type="error")
            return

        update_status("PREPROCESSING", f"Data split done. Train shape: {X_train.shape}", msg_type="success")

        # Model Zoo & Training
        update_status("TRAINING", f"Training models (Mode: {mode})...")
        
        # Handle Model Preference
        pref_model_instance = None
        user_preference_flag = False
        
        if mode == 'custom' and config['model'] and config['model'] != 'Auto' and not config.get('enable_ensemble'):
             from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
             from sklearn.linear_model import LogisticRegression, Ridge
             from xgboost import XGBClassifier, XGBRegressor
             
             user_preference_flag = True
             if config['model'] == 'RandomForest':
                 pref_model_instance = RandomForestClassifier() if zoo_problem_type == 'Classification' else RandomForestRegressor()
             elif config['model'] == 'XGBoost':
                 pref_model_instance = XGBClassifier() if zoo_problem_type == 'Classification' else XGBRegressor()
             elif config['model'] == 'Linear':
                 pref_model_instance = LogisticRegression() if zoo_problem_type == 'Classification' else Ridge()
        
        if config.get('enable_ensemble') and user_preference_flag:
            update_status("WARNING", "Ensemble requested. Ignoring single-model preference to ensure model diversity.", msg_type="warning")
            user_preference_flag = False
            pref_model_instance = None

        models = Model_zoo(zoo_problem_type, user_preferred_model=pref_model_instance, user_preference=user_preference_flag)
        
        # Pass update_status as callback
        leaderboard, fitted_models = model_training_evaluation(models, X_train, y_train, zoo_problem_type, X_test, y_test, log_callback=update_status)
        
        # --- NEW: Advanced Features (Tuning & Ensemble) ---
        from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, mean_absolute_error
        
        # 1. Hyperparameter Tuning
        if config.get("enable_tuning") and not leaderboard.empty:
            update_status("TUNING", "Running Hyperparameter Optimization (Optuna)...")
            try:
                # Tune the current best model
                best_model_name = leaderboard.iloc[0]['Model_Name']
                best_model_obj = fitted_models.get(best_model_name)
                
                if best_model_obj:
                    # Run tuning
                    tuned_model, best_params = auto_tune_model(
                        best_model_obj, X_train, y_train, zoo_problem_type, 
                        fast_mode=config.get("fast_train")
                    )
                    
                    # Evaluate Tuned Model
                    preds = tuned_model.predict(X_test)
                    name = f"{best_model_name}_Tuned"
                    
                    new_row = {"Model_Name": name}
                    if zoo_problem_type in ["Regression", "Time_series"]:
                        mse = mean_squared_error(y_test, preds)
                        new_row.update({
                            "RMSE": np.sqrt(mse),
                            "MAE": mean_absolute_error(y_test, preds),
                            "MSE": mse
                        })
                        leaderboard = pd.concat([leaderboard, pd.DataFrame([new_row])], ignore_index=True).sort_values(by="RMSE", ascending=True)
                    else:
                        new_row.update({
                            "Accuracy": accuracy_score(y_test, preds),
                            "Precision": precision_score(y_test, preds, average='weighted', zero_division=0)
                        })
                        leaderboard = pd.concat([leaderboard, pd.DataFrame([new_row])], ignore_index=True).sort_values(by="Accuracy", ascending=False)
                    
                    fitted_models[name] = tuned_model
                    update_status("TUNING", f"Optimization Complete! Best Params found.", msg_type="success")
            except Exception as e:
                update_status("TUNING", f"Tuning skipped due to error: {e}", msg_type="warning")

        # 2. Ensemble Methods
        if config.get("enable_ensemble"):
            # Determine which models to include in ensemble: user-selected or top performers
            method = config.get('ensemble_method', 'stacking')
            sel = config.get('selected_models')
            if sel and isinstance(sel, list) and len(sel) >= 2:
                chosen_models = {k: v for k, v in fitted_models.items() if k in sel}
                update_status("ENSEMBLE", f"Building Ensemble from user-selected models: {sel}", msg_type="info")
            else:
                # Fallback: take top-n from leaderboard
                top_names = leaderboard['Model_Name'].tolist()[:3]
                chosen_models = {k: fitted_models[k] for k in top_names if k in fitted_models}
                update_status("ENSEMBLE", f"Building Ensemble from top models: {top_names}", msg_type="info")

            if len(chosen_models) >= 2:
                update_status("ENSEMBLE", f"Creating ensemble (method={method})...", msg_type="info")
                try:
                    ensemble = auto_ensemble(
                        chosen_models, X_train, y_train, X_test, y_test,
                        problem_type=zoo_problem_type, top_n=len(chosen_models), method=method
                    )

                    # Evaluate Ensemble
                    preds = ensemble.predict(X_test)
                    name = f"Ensemble_{method.capitalize()}"

                    new_row = {"Model_Name": name}
                    if zoo_problem_type in ["Regression", "Time_series"]:
                        mse = mean_squared_error(y_test, preds)
                        new_row.update({
                            "RMSE": np.sqrt(mse),
                            "MAE": mean_absolute_error(y_test, preds),
                            "MSE": mse
                        })
                        leaderboard = pd.concat([leaderboard, pd.DataFrame([new_row])], ignore_index=True).sort_values(by="RMSE", ascending=True)
                    else:
                        new_row.update({
                            "Accuracy": accuracy_score(y_test, preds),
                            "Precision": precision_score(y_test, preds, average='weighted', zero_division=0)
                        })
                        leaderboard = pd.concat([leaderboard, pd.DataFrame([new_row])], ignore_index=True).sort_values(by="Accuracy", ascending=False)

                    fitted_models[name] = ensemble
                    update_status("ENSEMBLE", "Ensemble created successfully!", msg_type="success")
                except Exception as e:
                    update_status("ENSEMBLE", f"Ensemble creation failed: {e}", msg_type="warning")
            else:
                update_status("ENSEMBLE", f"Skipped: Need at least 2 models for ensemble, but found {len(chosen_models)}.", msg_type="warning")

        # Final Leaderboard Reporting
        for idx, row in leaderboard.iterrows():
             if zoo_problem_type in ["Regression", "Time_series"]:
                 stats = f"RMSE: {row['RMSE']:.4f}, MAE: {row['MAE']:.4f}, MSE: {row['MSE']:.4f}"
             else:
                 stats = f"Accuracy: {row.get('Accuracy', 0):.4f}, Precision: {row.get('Precision',0):.4f}"
             
             update_status("LEADERBOARD", f"Model: {row['Model_Name']} | {stats}", msg_type="info")

        if leaderboard.empty:
             update_status("ERROR", "No models trained successfully.", msg_type="error")
             with SessionLocal() as db:
                job = db.query(TrainingJob).filter(TrainingJob.id == config['job_id']).first()
                if job: job.status = "FAILED"; db.commit()
             return
             
        score_col = leaderboard.columns[1] # heuristic
        best_val = leaderboard.iloc[0][score_col]
        update_status("EVALUATION", f"Training done. Best {score_col}: {best_val}")
        
        # Save leaderboard to status for frontend chart
        global PIPELINE_STATUS
        PIPELINE_STATUS["leaderboard"] = leaderboard.to_dict(orient='records')

        # Model Registry
        update_status("REGISTRY", "Registering best model...")
        if os.path.exists("best_model.pkl"):
            best_model = joblib.load("best_model.pkl")
            feature_schema = {
                "columns": list(X_train.columns) if hasattr(X_train, "columns") else "numpy_array",
                "problem_type": zoo_problem_type,
                "target_col": target_column,
                "datetime_col": dt_col if zoo_problem_type == "Time_series" else None
            }
            # Ensure leaderboard has data before registering
            if not leaderboard.empty:
                metrics_dict = leaderboard.iloc[0].to_dict()
                # Save recent history as context for TS lag generation
                context_df = df.tail(50) if zoo_problem_type == "Time_series" else None
                registry_path = model_registry(best_model, metrics_dict, config, feature_schema, context_df=context_df)
            else:
                update_status("ERROR", "Leaderboard empty. Cannot register model.")
            
            # Validation Chart Data (Subset for frontend)
            try:
                # Best Model preds
                val_preds = best_model.predict(X_test)
                y_test_list = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
                p_list = val_preds.tolist() if hasattr(val_preds, "tolist") else list(val_preds)
                
                validations = {
                    "actual": y_test_list[:100],
                    "predicted": p_list[:100]
                }
                
                # Baseline Model preds
                baseline_obj = fitted_models.get('baseline')
                if baseline_obj:
                    b_preds = baseline_obj.predict(X_test)
                    validations["baseline"] = b_preds.tolist()[:100] if hasattr(b_preds, "tolist") else list(b_preds)[:100]
                
                PIPELINE_STATUS["validations"] = validations
            except Exception as ve:
                print(f"Validation data error: {ve}")

            update_status("DEPLOYMENT", "Updating live model...")
            load_global_model()
            update_status("COMPLETED", "Process Finished.")
            
            with SessionLocal() as db:
                job = db.query(TrainingJob).filter(TrainingJob.id == config['job_id']).first()
                if job: job.status = "COMPLETED"; db.commit()
        else:
            update_status("ERROR", "Model file not found.")
            with SessionLocal() as db:
                job = db.query(TrainingJob).filter(TrainingJob.id == config['job_id']).first()
                if job: job.status = "FAILED"; db.commit()

    except Exception as e:
        update_status("ERROR", f"Pipeline failed: {str(e)}")
        with SessionLocal() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == config['job_id']).first()
            if job: job.status = "FAILED"; db.commit()
        import traceback
        traceback.print_exc()

@app.get("/models")
def list_models():
    """Returns a list of all registered models with scores for model selection."""
    if not os.path.exists("model_registry"):
        return []
    
    models = []
    try:
        versions = sorted(os.listdir("model_registry"), reverse=True)
        for v in versions:
            v_dir = os.path.join("model_registry", v)
            if not os.path.isdir(v_dir): continue
            
            meta_path = os.path.join(v_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    metrics = meta.get("performance_metrics", {})
                    # Get the primary score (first metric)
                    score_val = "N/A"
                    if metrics:
                         first_key = list(metrics.keys())[0]
                         score_val = f"{metrics[first_key]:.4f}" if isinstance(metrics[first_key], (int, float)) else str(metrics[first_key])

                    schema = meta.get("feature_schema", {})
                    models.append({
                        "id": v,
                        "type": meta.get("model_type"),
                        "date": meta.get("registration_date"),
                        "problem": schema.get("problem_type"),
                        "datetime_col": schema.get("datetime_col"),
                        "columns": schema.get("columns"),
                        "score": score_val
                    })
    except Exception as e:
        print(f"List Models Error: {e}")
    return models

# ============================================================================
# NEW ADVANCED ENDPOINTS
# ============================================================================

@app.post("/validate-data")
async def validate_dataset(file: UploadFile = File(...)):
    """
    Comprehensive data quality validation
    Returns detailed report with quality score and recommendations
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Run validation
        validation_report = validate_data(df)
        
        return {
            "status": "success",
            "quality_score": validation_report["quality_score"],
            "basic_info": validation_report["basic_info"],
            "missing_values": validation_report["missing_values"],
            "duplicates": validation_report["duplicates"],
            "outliers": validation_report["outliers"],
            "recommendations": validation_report["recommendations"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/explain/{version_id}")
async def explain_model_prediction(version_id: str, input_data: PredictionInput):
    """
    Get AI explanation for model predictions using SHAP
    Shows which features contributed most to the prediction
    """
    try:
        # Load model and data
        v_path = os.path.join("model_registry", version_id)
        if not os.path.exists(v_path):
            raise HTTPException(status_code=404, detail="Model version not found")
        
        model = joblib.load(os.path.join(v_path, "AutoML_Model.pkl"))
        
        # Load training data for background
        with open(os.path.join(v_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Get feature schema
        schema = metadata.get("feature_schema", {})
        expected_cols = schema.get("columns", [])
        
        # Prepare input
        X_input = pd.DataFrame(input_data.data)
        X_input = X_input.reindex(columns=expected_cols).fillna(0)
        
        # Load sample of training data for SHAP background
        context_file = os.path.join(v_path, "context.csv")
        if os.path.exists(context_file):
            X_train = pd.read_csv(context_file)
            target_col = schema.get("target_col")
            if target_col and target_col in X_train.columns:
                X_train = X_train.drop(columns=[target_col])
            X_train = X_train.reindex(columns=expected_cols).fillna(0).head(100)
        else:
            # Use input as background if no context available
            X_train = X_input
        
        # Create explainer
        problem_type = schema.get("problem_type", "Classification")
        explainer = create_explainer(model, X_train, problem_type)
        
        # Get explanation
        explanation = explainer.explain_prediction(X_input.iloc[0])
        
        # Get global importance
        global_importance = explainer.get_global_feature_importance(X_train, max_samples=50)
        
        return {
            "prediction_explanation": explanation,
            "global_feature_importance": global_importance,
            "model_version": version_id,
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/monitoring/dashboard")
def get_monitoring_dashboard():
    """
    Get monitoring dashboard with all model health metrics
    """
    try:
        health_data = get_all_model_health()
        return health_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard fetch failed: {str(e)}")


@app.get("/monitoring/{version_id}")
def get_model_monitoring(version_id: str):
    """
    Get detailed monitoring metrics for a specific model
    """
    try:
        monitor = ModelMonitor(version_id)
        dashboard = monitor.get_dashboard_data()
        
        return dashboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monitoring fetch failed: {str(e)}")

@app.get("/feature-importance/{version_id}")
def get_feature_importance(version_id: str, method: str = "shap"):
    """
    Get feature importance using SHAP or built-in methods
    """
    try:
        v_path = os.path.join("model_registry", version_id)
        if not os.path.exists(v_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = joblib.load(os.path.join(v_path, "AutoML_Model.pkl"))
        
        with open(os.path.join(v_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        schema = metadata.get("feature_schema", {})
        expected_cols = schema.get("columns", [])
        
        feature_importance = {}
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(expected_cols, importances.tolist()))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        elif method == "shap":
            # Use SHAP for model-agnostic importance
            context_file = os.path.join(v_path, "context.csv")
            if os.path.exists(context_file):
                X_sample = pd.read_csv(context_file).head(100)
                target_col = schema.get("target_col")
                if target_col and target_col in X_sample.columns:
                    X_sample = X_sample.drop(columns=[target_col])
                X_sample = X_sample.reindex(columns=expected_cols).fillna(0)
                
                explainer = create_explainer(model, X_sample, schema.get("problem_type", "Classification"))
                global_imp = explainer.get_global_feature_importance(X_sample)
                
                feature_importance = global_imp.get("global_importance", {})
        
        return {
            "model_version": version_id,
            "feature_importance": feature_importance,
            "method": "built-in" if hasattr(model, 'feature_importances_') else method,
            "top_10_features": dict(list(feature_importance.items())[:10])
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feature importance failed: {str(e)}")

@app.post("/predict")
def predict(input_data: PredictionInput):
    target_model = CURRENT_MODEL
    target_metadata = MODEL_METADATA
    target_context = CONTEXT_DATA

    # If user specified a specific version, load it on the fly
    if input_data.version_id:
        v_path = os.path.join("model_registry", input_data.version_id)
        if os.path.exists(v_path):
            try:
                # Load bits from the specified version folder
                model_file = os.path.join(v_path, "AutoML_Model.pkl")
                if os.path.exists(model_file):
                    target_model = joblib.load(model_file)
                
                meta_file = os.path.join(v_path, "metadata.json")
                if os.path.exists(meta_file):
                    with open(meta_file, 'r') as f:
                        target_metadata = json.load(f)
                
                context_file = os.path.join(v_path, "context.csv")
                if os.path.exists(context_file):
                    target_context = pd.read_csv(context_file)
            except Exception as le:
                raise HTTPException(status_code=500, detail=f"Failed to load model version {input_data.version_id}: {str(le)}")

    if target_model is None or target_metadata is None:
        raise HTTPException(status_code=503, detail="No compatible model found or loaded.")
    
    try:
        df = pd.DataFrame(input_data.data)
        schema = target_metadata.get("feature_schema", {})
        prob_type = schema.get("problem_type")
        expected_cols = schema.get("columns", [])

        if prob_type == "Time_series":
            dt_col = schema.get("datetime_col")
            target_col = schema.get("target_col")
            
            if target_context is not None:
                # Merge with context to compute lags correctly
                full_df = pd.concat([target_context, df], ignore_index=True)
                processed = time_series_preprocessing_final(full_df, dt_col, target_col)
                # Take only the rows that correspond to our new input data
                X_pred = processed.tail(len(df))
            else:
                X_pred = time_series_preprocessing_final(df, dt_col, target_col)
            
            # CRITICAL: Ensure we only pass the columns the model expects and in same order
            # Filter out any columns not in expected_cols
            X_input = X_pred.reindex(columns=expected_cols).fillna(0)
        else:
            # Reindex ensures columns match training exactly
            X_input = df.reindex(columns=expected_cols).fillna(0)

        prediction = target_model.predict(X_input)
        return {"predictions": prediction.tolist() if hasattr(prediction, "tolist") else list(prediction)}
    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/forecast/{version_id}/{steps}")
def forecast(version_id: str, steps: int):
    """Performs recursive multi-step forecasting for Time Series models."""
    v_path = os.path.join("model_registry", version_id)
    if not os.path.exists(v_path):
        raise HTTPException(status_code=404, detail="Model version not found")

    try:
        # 1. Load context and metadata
        with open(os.path.join(v_path, "metadata.json"), 'r') as f:
            meta = json.load(f)
        
        schema = meta.get("feature_schema", {})
        if schema.get("problem_type") != "Time_series":
            raise HTTPException(status_code=400, detail="Forecasting is only available for Time Series models.")

        model = joblib.load(os.path.join(v_path, "AutoML_Model.pkl"))
        context_df = pd.read_csv(os.path.join(v_path, "context.csv"))
        
        dt_col = schema.get("datetime_col")
        target_col = schema.get("target_col")
        expected_cols = schema.get("columns", [])

        # 2. Recursive Forecasting Logic
        context_df[dt_col] = pd.to_datetime(context_df[dt_col])
        history = context_df.copy()
        predictions = []

        for i in range(steps):
            # Create next row (assuming daily frequency for simplicity, otherwise detect freq)
            last_date = history[dt_col].max()
            next_date = last_date + pd.Timedelta(days=1)
            
            # Create a placeholder row
            new_row = {dt_col: next_date}
            for col in history.columns:
                if col not in [dt_col, target_col]:
                    new_row[col] = history[col].iloc[-1] # Propagate other features
            
            # Append to history
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
            
            # Apply preprocessing to generate lags/rolling for the new row
            processed = time_series_preprocessing_final(history, dt_col, target_col)
            
            # Extract features for the latest row
            latest_features = processed.tail(1).reindex(columns=expected_cols).fillna(0)
            
            # Predict
            pred_val = model.predict(latest_features)[0]
            predictions.append({"date": str(next_date.date()), "value": float(pred_val)})
            
            # Update the last row in history with the prediction to feed into next lag
            history.loc[history.index[-1], target_col] = pred_val

        return {"forecast": predictions}

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{version_id}")
def download_model(version_id: str):
    """Allows downloading the trained .pkl model file."""
    print(f"DEBUG: Download request for version: {version_id}")
    v_path = os.path.join("model_registry", version_id)
    model_file = os.path.join(v_path, "AutoML_Model.pkl")
    
    print(f"DEBUG: Checking path: {os.path.abspath(model_file)}")
    if os.path.exists(model_file):
        print(f"DEBUG: File found. Serving download.")
        return FileResponse(
            path=model_file,
            filename=f"AutoML_Model_{version_id}.pkl",
            media_type='application/octet-stream'
        )
    print(f"DEBUG: File NOT found at {os.path.abspath(model_file)}")
    raise HTTPException(status_code=404, detail=f"Model file not found for version {version_id}")

@app.post("/monitor")
def monitor(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        prod_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        train_files = [f for f in os.listdir("sample_data") if f.endswith(".csv")]
        if not train_files: raise HTTPException(404, "No training data")
        train_df = pd.read_csv(os.path.join("sample_data", train_files[0]))
        is_drifted, report = check_data_drift(train_df, prod_df)
        return {"drift_detected": bool(is_drifted), "report": {k:str(v) for k,v in report.items()}}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
