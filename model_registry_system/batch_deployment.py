import pandas as pd
import joblib
import os
import json

def batch_prediction_pipeline(input_file, model_version=None):
    # 1. Load Model (Latest agar version specify nahi hai)
    registry_dir = "model_registry"
    version = model_version if model_version else sorted(os.listdir(registry_dir))[-1]
    model_path = f"{registry_dir}/{version}/AutoML_Model.pkl"
    meta_path = f"{registry_dir}/{version}/metadata.json"
    
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # 2. Load New Data
    df = pd.read_csv(input_file)
    
    # 3. Apply stored preprocessing (Yahan aapka cleaning module call hoga)
    # Note: feature_info metadata se lena hai taaki columns mismatch na hon
    # df_processed = clean_and_transform(df, meta['feature_schema'])

    # 4. Predict
    predictions = model.predict(df)
    df['predictions'] = predictions
    
    output_path = f"outputs/predictions_{version}.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Batch Predictions saved to {output_path}")
    return output_path
