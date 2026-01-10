import os
import json
import joblib
from datetime import datetime

def model_registry(best_model, metrics, params, feature_info, model_name="AutoML_Model", context_df=None):
    """
    Model ko versioning aur metadata ke saath register karna.
    """
    # 1. Versioning Logic (Timestamp based)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_id = f"v_{timestamp}"
    registry_path = f"model_registry/{version_id}"
    
    if not os.path.exists(registry_path):
        os.makedirs(registry_path)

    # 2. Metadata taiyar karna
    metadata = {
        "model_id": version_id,
        "model_type": str(type(best_model).__name__),
        "registration_date": str(datetime.now()),
        "performance_metrics": metrics,
        "best_hyperparameters": params,
        "feature_schema": feature_info,
        "status": "Production_Ready"
    }

    # 3. Saving the Bundle
    model_file = os.path.join(registry_path, f"{model_name}.pkl")
    joblib.dump(best_model, model_file)

    meta_file = os.path.join(registry_path, "metadata.json")
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Save context data if available (for Time Series lags)
    if context_df is not None:
        context_df.to_csv(os.path.join(registry_path, "context.csv"), index=False)

    print(f"✅ Model Registered under Version: {version_id}")
    print(f"📂 Path: {registry_path}")
    
    return registry_path

def get_latest_model():
    """Registry se sabse naya model uthana (Deployment ke liye)"""
    if not os.path.exists("model_registry"):
        return None
    all_versions = sorted(os.listdir("model_registry"), reverse=True)
    if not all_versions:
        return None
    latest_path = os.path.join("model_registry", all_versions[0])
    return latest_path
