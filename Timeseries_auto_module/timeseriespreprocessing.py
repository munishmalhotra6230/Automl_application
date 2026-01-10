import pandas as pd
import numpy as np
import holidays
import os

def time_series_preprocessing_final(df, datetime_col, target_col, freq=None, country='IN'):
    """
    Professional Time Series Pipeline: Robust date handling, feature engineering, 
    and automatic handling of additional numeric/categorical features.
    """
    # 1. Date Formatting & Robust Conversion
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Drop rows where date conversion failed
    df = df.dropna(subset=[datetime_col])
    
    # 2. Sorting and Indexing
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    df = df.set_index(datetime_col)

    # 3. Handle Other Features (Encoding & Imputation)
    # Identify non-target, non-date columns
    other_cols = [c for c in df.columns if c != target_col]
    
    for col in other_cols:
        # Numeric: Simple Impute
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].ffill().bfill().fillna(0)
        # Categorical: Label Encode & Impute
        else:
            df[col] = df[col].astype(str).fillna('Unknown')
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # 4. Feature Engineering - Date Components
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['hour'] = df.index.hour
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)

    # 5. Cyclical Features (Fourier)
    df['sin_month'] = np.sin(2 * np.pi * df['month']/12)
    df['cos_month'] = np.cos(2 * np.pi * df['month']/12)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_week']/7)

    # 6. Holiday Detection
    try:
        custom_holidays = holidays.CountryHoliday(country)
        df['is_holiday'] = pd.Series(df.index).apply(lambda x: 1 if x in custom_holidays else 0).values
    except:
        df['is_holiday'] = 0

    # 7. Lag Features (Critical)
    lags = [1, 2, 3, 7, 14]
    for lag in lags:
        if len(df) > lag:
            df[f'lag_{lag}'] = df[target_col].shift(lag)

    # 8. Rolling Statistics
    windows = [7, 14]
    for w in windows:
        if len(df) > w:
            df[f'rolling_mean_{w}'] = df[target_col].shift(1).rolling(window=w).mean()
            df[f'rolling_std_{w}'] = df[target_col].shift(1).rolling(window=w).std()

    # 9. Differencing (Shifted by 1 to prevent data leakage)
    df['target_diff'] = df[target_col].diff(1).shift(1)
    
    # Optional: Percentage Change (also useful, also shifted)
    df['target_pct_change'] = df[target_col].pct_change(1).shift(1)

    # 10. Final Cleaning
    df = df.dropna()
    
    return df.reset_index()

def time_series_split(df, test_size=0.2):
    """
    Ensures sequential splitting for time series data.
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test
