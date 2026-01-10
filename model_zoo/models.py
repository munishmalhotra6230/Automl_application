import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,accuracy_score,precision_score,recall_score,confusion_matrix


def Model_zoo(problem_type, user_preferred_model=None, baseline=None, user_preference=False):
    """
    Optimized Model Zoo for 2026: Fast Training & High Performance.
    """
    
    if user_preference == False:
        # 1. CLASSIFICATION MODELS
        if problem_type == "Classification":
            models = {
                "baseline": LogisticRegression(max_iter=1000, n_jobs=-1),
                "Tree": RandomForestClassifier(n_jobs=-1),
                "Boosting": XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='hist', n_jobs=-1),
            }
            
        # 2. REGRESSION MODELS
        elif problem_type == "Regression":
            models = {
                "baseline": Ridge(),
                "Tree": RandomForestRegressor(n_jobs=-1),
                "Boosting": XGBRegressor(tree_method='hist', n_jobs=-1),
            }
            
        # 3. TIME SERIES MODELS
        elif problem_type == "Time_series":
            models = {
                "baseline": Ridge(), 
                "tree": RandomForestRegressor(n_jobs=-1),
                "boosting": XGBRegressor(tree_method='hist', n_jobs=-1)
            }
            
        # 4. ANOMALY DETECTION MODELS
        elif problem_type == "Anomaly":
            models = {
                "baseline": IsolationForest(contamination='auto', n_jobs=-1),
                "tree": LocalOutlierFactor(novelty=True, n_jobs=-1)
            }
            
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
            
    else:
        # USER PREFERENCE LOGIC
        if user_preferred_model is None:
            raise ValueError("User preference is True but no model was provided.")
            
        # Attempt to inject n_jobs=-1 if not already set, for speed
        if hasattr(user_preferred_model, 'n_jobs'):
            try: user_preferred_model.set_params(n_jobs=-1)
            except: pass

        if problem_type == "Classification":
            default_baseline = LogisticRegression(n_jobs=-1)
        elif problem_type == "Regression":
            default_baseline = Ridge()
        elif problem_type == "Time_series":
            default_baseline = Ridge()
        else:
            default_baseline = LogisticRegression(n_jobs=-1)

        models = {
            "baseline": baseline if baseline else default_baseline, 
            "user_specified": user_preferred_model
        }
    
    return models
import pandas as pd
import numpy as np
import joblib # Model save karne ke liye
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             accuracy_score, precision_score, 
                             confusion_matrix, silhouette_score)

def model_training_evaluation(models, xtrain, ytrain, problem_type, xtest, ytest, log_callback=None):
    results_list = []
    fitted_models_dict = {}
    best_model = None
    prob_type_lower = problem_type.lower()
    
    if "classification" in prob_type_lower:
        best_score = -np.inf
    else:
        best_score = np.inf

    for name, model in models.items():
        try:
            msg = f"🚀 Training {name}..."
            print(msg)
            if log_callback: log_callback("TRAINING", msg)
            
            # 1. Training Logic
            model.fit(xtrain, ytrain)
            preds = model.predict(xtest)
            current_fitted_model = model
            fitted_models_dict[name] = current_fitted_model

            # 2. Evaluation Logic
            res = {"Model_Name": name}
            
            if "regression" in prob_type_lower or "time_series" in prob_type_lower:
                res["MSE"] = mean_squared_error(ytest, preds)
                res["RMSE"] = np.sqrt(res["MSE"])
                res["MAE"] = mean_absolute_error(ytest, preds)
                # Best model selection (Lower RMSE is better)
                if res["RMSE"] < best_score:
                    best_score = res["RMSE"]
                    best_model = current_fitted_model

            elif "classification" in prob_type_lower:
                res["Accuracy"] = accuracy_score(ytest, preds)
                res["Precision"] = precision_score(ytest, preds, average='weighted', zero_division=0)
                # Best model selection (Higher Accuracy is better)
                if res["Accuracy"] > best_score:
                    best_score = res["Accuracy"]
                    best_model = current_fitted_model

            elif "anomaly" in prob_type_lower:
                # Anomaly mein Silhouette Score use hota hai (Range -1 to 1)
                score = silhouette_score(xtrain, model.fit_predict(xtrain))
                res["Silhouette_Score"] = score
                if score > best_score:
                    best_score = score
                    best_model = current_fitted_model

            results_list.append(res)

        except Exception as e:
            print(f"❌ Error in {name}: {e}")

    # 3. Create Leaderboard
    leaderboard = pd.DataFrame(results_list)
    if not leaderboard.empty:
        # Sort by score (best first)
        if "classification" in prob_type_lower:
            leaderboard = leaderboard.sort_values(by=leaderboard.columns[1], ascending=False)
        else:
            leaderboard = leaderboard.sort_values(by=leaderboard.columns[1], ascending=True)

    # 4. Save Best Model
    if best_model:
        joblib.dump(best_model, 'best_model.pkl')
        msg = "✅ Best Model Saved as 'best_model.pkl'"
        print(f"\n{msg}")
        if log_callback: log_callback("REGISTRY", msg, msg_type="success")
    else:
        msg = "⚠️ No best model could be determined."
        print(f"\n{msg}")
        if log_callback: log_callback("ERROR", msg, msg_type="warning")

    return leaderboard, fitted_models_dict



        

    







