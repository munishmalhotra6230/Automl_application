import pandas as pd 
import numpy as np 
import json
import requests
import pandas as pd
import io
import os 
def data_loader(file_path, encoding="utf-8", schema_path="schema.json"):
    df = pd.read_csv(file_path, encoding=encoding)
    
    # Validation Logic
    try:
        with open(schema_path, 'r') as f:
            base_schema = json.load(f)
        
        current_columns = set(df.columns)
        base_columns = set(base_schema.keys())

        # 1. Column Drift Check (Missing or Extra Columns)
        if current_columns != base_columns:
            missing = base_columns - current_columns
            extra = current_columns - base_columns
            print(f"ALARM: Schema Mismatch! Missing: {missing}, Extra: {extra}")
            # Aap yahan system ko stop bhi kar sakte hain agar critical ho

        # 2. Type Drift Check
        for col, expected_type in base_schema.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    print(f"WARNING: Type Drift in {col}! Expected {expected_type}, got {actual_type}")
                    
    except FileNotFoundError:
        print("Initial run: No base schema found to validate.")
        
    return df
def validate_schema(df, schema_path="schema.json"):
    """
    Kissi bhi DataFrame ko existing schema se validate karta hai.
    """
    if not os.path.exists(schema_path):
        print("Initial run: No base schema found. Saving current as base.")
        save_base_schema(df, schema_path)
        return True, "Schema Initialized"

    with open(schema_path, 'r') as f:
        base_schema = json.load(f)
    
    current_columns = set(df.columns)
    base_columns = set(base_schema.keys())
    
    report = []
    is_valid = True

    # 1. Column Check
    if current_columns != base_columns:
        missing = base_columns - current_columns
        extra = current_columns - base_columns
        report.append(f"Mismatch! Missing: {missing}, Extra: {extra}")
        is_valid = False

    # 2. Type Check
    for col, expected_type in base_schema.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                report.append(f"Type Drift in {col}: Expected {expected_type}, got {actual_type}")
                is_valid = False
                
    return is_valid, report


def api_data_loader(api_url, headers=None, params=None):
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=10)
        response.raise_for_status() 
        
        # FIX: Nested JSON handling (pd.json_normalize)
        if 'application/json' in response.headers.get('Content-Type', '') or api_url.endswith('.json'):
            data = response.json()
            if isinstance(data, dict):
                df = pd.json_normalize(data) # Complex APIs ke liye best hai
            else:
                df = pd.DataFrame(data)
        else:
            df = pd.read_csv(io.StringIO(response.text))
            
        print(f"Successfully loaded {len(df)} rows from API.")
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def universal_data_loader(source, source_type="file", schema_path="schema.json", **kwargs):
    if source_type == "file":
        df = data_loader(source)
    elif source_type == "api":
        df = api_data_loader(source, **kwargs)
    else:
        raise ValueError("Invalid source_type")

    # API ho ya File, dono yahan validate honge
    is_valid, report = validate_schema(df, schema_path)
    if not is_valid:
        print(f"⚠️ SCHEMA VALIDATION FAILED: {report}")
    else:
        print("✅ Schema Validation Passed.")
    return df

def save_base_schema(df, schema_path="schema.json"):
    # Har column ka name aur uska expected data type save karein
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    with open(schema_path, 'w') as f:
        json.dump(schema, f)
    print("Base Schema Saved Successfully!")

def data_stats(df):# stats_of_data
    dict_meta_data_of_df = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "nan_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()), # .duplicated() fix kiya
        "column_name": df.columns.tolist()
    }
    stats_summary = df.describe(include='all').to_dict() # include='all' se categories bhi aayengi
    health_score = 100 - (df.isna().sum().sum() / df.size * 100)
    return dict_meta_data_of_df, stats_summary, health_score
def feature_categorizer(df):
    categorical_cols = []
    numerical_cols = []
    datetime_cols = []
    id_cols = [] # High cardinality columns jo model ke kaam ke nahi hain

    for col in df.columns:
        # 1. Check for Datetime
        # Agar string format mein date hai toh convert karke check karein
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
                continue
            except:
                pass

        # 2. Check for ID Columns (High Cardinality)
        # Agar unique values total rows ke 95% se zyada hain aur wo object/int hai
        if df[col].nunique() / len(df) > 0.95:
            id_cols.append(col)
            continue

        # 3. Check for Categorical
        # Agar unique values kam hain (e.g., < 20) toh wo category hai
        if df[col].nunique() < 20 or df[col].dtype == 'object':
            categorical_cols.append(col)
        
        # 4. Check for Numerical
        else:
            numerical_cols.append(col)

    return {
        "categorical": categorical_cols,
        "numerical": numerical_cols,
        "datetime": datetime_cols,
        "id": id_cols
    }

def column_dtype_transformation(df, dtype_mapping):
    # dtype_mapping should be a dict: {"col1": "int32", "col2": "float32"}
    try:
        for col, new_type in dtype_mapping.items():
            if col in df.columns:
                df[col] = df[col].astype(new_type)
    except Exception as e:
        print(f"Error in transformation: {e}")
    return df
   

def user_guided_column_removal(_usercolumns, df, auto=False, missing_percent=0.7):
    if not auto:
        # Note: drop(inplace=True) returns None; it's safer to return the df
        df.drop(columns=_usercolumns, inplace=True)
        return df
    else:
        # Calculate threshold: e.g., if 70% or more is missing, drop it
        limit = len(df) * missing_percent
        for column in df.columns:
            if df[column].isna().sum() >= limit:
                df.drop(columns=[column], inplace=True)
        return df
    
def removing_nan_duplicated_values(df, feature_info, user_conformation_drop_nan=False):
    df.drop_duplicates(inplace=True)
    
    # Agar data bahut bada hai toh rows drop karna safe hai
    if len(df) > 10000 or user_conformation_drop_nan:
        df.dropna(inplace=True)
        return df

    # Smart Imputation based on Feature Categorizer
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if col in feature_info['categorical']:
                df[col] = df[col].fillna(df[col].mode()[0])
            elif col in feature_info['numerical']:
                # Advanced: Check skewness
                if abs(df[col].skew()) > 0.75:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mean())
    return df
def optimize_memory(df):
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer' if 'int' in str(df[col].dtype) else 'float')
    return df
def auto_encode(df, feature_info):
    for col in feature_info['categorical']:
        if df[col].nunique() <= 5:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            df[col] = df[col].astype('category').cat.codes
    return df
def handle_outliers(df, feature_info, strategy="clip"):
    for col in feature_info['numerical']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bridge = Q1 - (IQR * 1.5)
        upper_bridge = Q3 + (IQR * 1.5)
        
        if strategy == "clip":
            # Outliers ko limit par set kar dena
            df[col] = np.clip(df[col], lower_bridge, upper_bridge)
        elif strategy == "remove":
            df = df[(df[col] >= lower_bridge) & (df[col] <= upper_bridge)]
    return df








    

    

