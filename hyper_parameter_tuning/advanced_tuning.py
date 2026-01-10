"""
Advanced Hyperparameter Optimization using Optuna
Supports all major model types with intelligent parameter suggestions
"""
import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedHyperparameterTuner:
    """
    Intelligent hyperparameter optimization for different model types
    """
    
    def __init__(self, model_type, problem_type, n_trials=50, timeout=300, fast_mode=False):
        """
        Args:
            model_type: str - 'xgboost', 'randomforest', 'lightgbm', 'catboost', etc.
            problem_type: str - 'Classification', 'Regression', 'Time_series'
            n_trials: int - Number of optimization trials
            timeout: int - Max seconds for optimization
            fast_mode: bool - Use fewer trials for speed
        """
        self.model_type = model_type.lower()
        self.problem_type = problem_type
        self.n_trials = 20 if fast_mode else n_trials
        self.timeout = timeout
        self.best_params = None
        self.best_score = None
        
    def get_param_space(self, trial):
        """Define parameter search space for each model type"""
        
        if 'xgboost' in self.model_type or 'xgb' in self.model_type:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.01, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
        elif 'lightgbm' in self.model_type or 'lgb' in self.model_type:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
        elif 'catboost' in self.model_type or 'cat' in self.model_type:
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            }
            
        elif 'randomforest' in self.model_type or 'rf' in self.model_type:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
            
        elif 'logistic' in self.model_type:
            params = {
                'C': trial.suggest_float('C', 1e-3, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
            }
            
        elif 'ridge' in self.model_type or 'lasso' in self.model_type:
            params = {
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
            }
            
        else:
            # Generic parameters for unknown models
            params = {
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
            }
            
        return params
    
    def optimize(self, model, X_train, y_train, cv=3, scoring=None):
        """
        Run hyperparameter optimization
        
        Args:
            model: sklearn-like model instance
            X_train: Training features
            y_train: Training labels
            cv: Cross-validation folds
            scoring: Scoring metric (auto-selected if None)
        
        Returns:
            best_model: Model with best parameters
            best_params: Dictionary of best parameters
            study: Optuna study object
        """
        
        # Auto-select scoring metric
        if scoring is None:
            if self.problem_type in ['Classification', 'Binary_classification', 'Multi_classification']:
                scoring = 'accuracy'
            else:  # Regression or Time_series
                scoring = 'neg_mean_squared_error'
        
        def objective(trial):
            # Get parameter suggestions
            params = self.get_param_space(trial)
            
            # Set parameters on model
            try:
                model.set_params(**params)
            except:
                # Some params might not be compatible
                pass
            
            # Cross-validation score
            try:
                scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs=-1
                )
                return scores.mean()
            except Exception as e:
                # Return poor score if model fails
                return -1e10 if 'neg_' in scoring else 0
        
        # Create study
        sampler = TPESampler(seed=42)
        direction = 'maximize' if 'neg_' not in scoring else 'minimize'
        study = optuna.create_study(direction=direction, sampler=sampler)
        
        # Optimize with progress callback
        print(f"🔍 Starting hyperparameter optimization ({self.n_trials} trials)...")
        
        study.optimize(
            objective, 
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
            n_jobs=1  # Parallel trials can cause issues
        )
        
        # Get best results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"✅ Best score: {self.best_score:.4f}")
        print(f"📊 Best parameters: {self.best_params}")
        
        # Set best parameters and train final model
        model.set_params(**self.best_params)
        model.fit(X_train, y_train)
        
        return model, self.best_params, study
    
    def get_feature_importance_from_study(self, study):
        """Get parameter importance from optimization study"""
        try:
            from optuna.importance import get_param_importances
            importances = get_param_importances(study)
            return importances
        except:
            return {}


def auto_tune_model(model, X_train, y_train, problem_type, fast_mode=False):
    """
    Convenience function for quick hyperparameter tuning
    
    Args:
        model: Model instance to tune
        X_train: Training features
        y_train: Training labels
        problem_type: 'Classification', 'Regression', or 'Time_series'
        fast_mode: Use faster but less thorough optimization
    
    Returns:
        tuned_model: Model with optimized parameters
        best_params: Best parameters found
    """
    
    # Detect model type
    model_name = type(model).__name__.lower()
    
    # Create tuner
    tuner = AdvancedHyperparameterTuner(
        model_type=model_name,
        problem_type=problem_type,
        n_trials=20 if fast_mode else 50,
        timeout=180 if fast_mode else 600,
        fast_mode=fast_mode
    )
    
    # Run optimization
    tuned_model, best_params, study = tuner.optimize(model, X_train, y_train)
    
    return tuned_model, best_params
