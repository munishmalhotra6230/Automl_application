import pandas as pd
import numpy as np
import optuna
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV, cross_val_score

def integrated_tuner(model, xtrain, ytrain, problem_type, data_size_threshold=50000):
    """
    Data size ke basis par Halving (Bisection) ya Bayesian (Optuna) choose karta hai.
    """
    
    rows = xtrain.shape[0]
    
    # 1. LARGE DATA -> HALVING SEARCH (Speed focus)
    if rows > data_size_threshold:
        print(f"⚡ Large Dataset ({rows} rows) detected. Using Halving Search (Bisection)...")
        
        # Example Param Grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9]
        }
        
        search = HalvingRandomSearchCV(
            estimator=model,
            param_distributions=param_grid,
            factor=3, # Bisection factor
            cv=3,
            n_jobs=-1
        )
        search.fit(xtrain, ytrain)
        return search.best_estimator_, search.best_params_

    # 2. SMALL/MEDIUM DATA -> BAYESIAN OPTIMIZATION (Accuracy focus)
    else:
        print(f"🧠 Optimized Dataset ({rows} rows) detected. Using Bayesian Optimization (Optuna)...")
        
        def objective(trial):
            # Dynamic Parameter Suggestion
            if "Classifier" in str(type(model)):
                lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
                depth = trial.suggest_int("max_depth", 3, 12)
                model.set_params(learning_rate=lr, max_depth=depth)
            else:
                alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
                model.set_params(alpha=alpha)
                
            return cross_val_score(model, xtrain, ytrain, n_jobs=-1, cv=3).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30) # 30 trials enough for learning
        
        model.set_params(**study.best_params)
        model.fit(xtrain, ytrain)
        return model, study.best_params
    
