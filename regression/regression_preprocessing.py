import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def Regression_preprocessing(df, target_column, ordinal=None, onehot=None, order=None, 
                             standard_scaler=None, minmax=None, robust=None, bulk=False, 
                             split_ratio=0.2, bulk_scaling_method="standard", column_config=None):
    
    # 0. Clean Target
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])

    # 0b. Handle Boolean Columns early (Imputer fix)
    for col in df.columns:
        if df[col].dtype == 'bool' or pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)

    # 1. Custom Granular Handling
    if column_config:
        drop_cols = [c for c, cfg in column_config.items() if cfg.get('handling') == 'drop']
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Extract manual encodings/scalings
        onehot = [c for c, cfg in column_config.items() if cfg.get('transform') == 'onehot' and c in df.columns]
        standard_scaler = [c for c, cfg in column_config.items() if cfg.get('transform') == 'standard' and c in df.columns]
        minmax = [c for c, cfg in column_config.items() if cfg.get('transform') == 'minmax' and c in df.columns]
        robust = [c for c, cfg in column_config.items() if cfg.get('transform') == 'robust' and c in df.columns]
        ordinal = [c for c, cfg in column_config.items() if cfg.get('transform') == 'ordinal' and c in df.columns]

        # Granular Imputation
        for col, cfg in column_config.items():
            if col in df.columns and cfg.get('impute') != 'none':
                method = cfg.get('impute')
                if method == 'mean': df[col] = df[col].fillna(df[col].mean())
                elif method == 'median': df[col] = df[col].fillna(df[col].median())
                elif method == 'mode': df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
    """
    AutoML Regression Pipeline: Encoding, Outlier-Safe Scaling, aur Target Transformation.
    """
    
    # 1. One-Hot Encoding
    if onehot:
        df = pd.get_dummies(data=df, columns=onehot, drop_first=True, dtype=int)
    
    # 2. Ordinal Encoding (Handling Unknowns for Production)
    if ordinal and order:
        oe = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=-1)
        for col in ordinal:
            df[[col]] = oe.fit_transform(df[[col]])

    # 3. Target Skewness Handling (2026 MLOps Best Practice)
    # Agar target skewed hai, toh Log Transformation model accuracy badha deta hai
    if abs(df[target_column].skew()) > 0.75:
        print(f"ℹ️ Target '{target_column}' is skewed. Applying Log Transformation.")
        df[target_column] = np.log1p(df[target_column])

    # 4. Split Data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Regression mein random split chalta hai (stratify nahi)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    # 5. Smart Imputation (Filling NaNs before Scaling)
    # Separate numeric and categorical for safe imputation
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()

    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])
        
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        # Cast to string to avoid dtype issues (Sklearn imputer likes strings for categorical)
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols].astype(str))
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols].astype(str))
    
    # 6. Scaling Logic with Passthrough
    if bulk:
        numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        if bulk_scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif bulk_scaling_method == "robust":
            scaler = RobustScaler() # Regression mein outliers ke liye robust best hai
        else:
            scaler = StandardScaler()

        ct = ColumnTransformer([
            ("bulk_scaler", scaler, numeric_cols)
        ], remainder='passthrough')
    else:
        transformers = []
        if standard_scaler: transformers.append(("standard", StandardScaler(), standard_scaler))
        if minmax: transformers.append(("minmax", MinMaxScaler(), minmax))
        if robust: transformers.append(("robust", RobustScaler(), robust))
        
        ct = ColumnTransformer(transformers, remainder='passthrough')

    # Final Arrays
    X_train_final = ct.fit_transform(X_train)
    X_test_final = ct.transform(X_test)

    return X_train_final, X_test_final, y_train, y_test
