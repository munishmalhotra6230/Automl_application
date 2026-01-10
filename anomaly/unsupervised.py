import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

def Anomaly_preprocessing(df, ordinal=None, onehot=None, order=None, 
                          standard_scaler=None, robust=None, bulk=True, 
                          use_pca=False, pca_components=0.95):
    """
    AutoML Anomaly Detection Pipeline: No target column, focused on feature scaling.
    """
    df_clean = df.copy()

    # 0. Handle Boolean Columns early (Imputer fix)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'bool':
            df_clean[col] = df_clean[col].astype(int)

    # 1. Categorical Encoding
    if onehot:
        df_clean = pd.get_dummies(data=df_clean, columns=onehot, drop_first=True, dtype=int)
    
    if ordinal and order:
        oe = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=-1)
        for col in ordinal:
            df_clean[[col]] = oe.fit_transform(df_clean[[col]])

    # 2. Handling Missing Values (Imputation)
    num_imputer = SimpleImputer(strategy='median')
    df_imputed = num_imputer.fit_transform(df_clean)
    df_clean = pd.DataFrame(df_imputed, columns=df_clean.columns)

    # 3. Scaling (Anomaly mein RobustScaler best hota hai)
    if bulk:
        numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
        # Anomaly detection ke liye RobustScaler default hona chahiye
        scaler = RobustScaler() 
        ct = ColumnTransformer([
            ("bulk_scaler", scaler, numeric_cols)
        ], remainder='passthrough')
    else:
        transformers = []
        if standard_scaler: transformers.append(("standard", StandardScaler(), standard_scaler))
        if robust: transformers.append(("robust", RobustScaler(), robust))
        ct = ColumnTransformer(transformers, remainder='passthrough')

    X_processed = ct.fit_transform(df_clean)

    # 4. Dimensionality Reduction (Optional but Recommended for Anomaly)
    # PCA se noise hat jata hai aur anomalies detect karna aasaan hota hai
    if use_pca:
        # pca_components=0.95 matlab 95% variance retain karo
        pca = PCA(n_components=pca_components)
        X_processed = pca.fit_transform(X_processed)
        print(f"ℹ️ PCA applied. Features reduced to {X_processed.shape[1]} components.")

    return X_processed

# --- Usage Example ---
# X_final = Anomaly_preprocessing(df, bulk=True, use_pca=True)
