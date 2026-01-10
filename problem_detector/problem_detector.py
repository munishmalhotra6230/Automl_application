import pandas as pd
import json
import os
import numpy as np 

# problem detection 
def user_handled_problem(df, target_column=None, datetime_col=None):
    if target_column is None or target_column not in df.columns:
        return "Unsupervised_learning"
    
    target_column_dtype = df[target_column].dtype
    unique = df[target_column].nunique()
    
    # Check for datetime: either explicitly provided or automatically detected
    if datetime_col and datetime_col in df.columns:
        return "Time_series"
        
    date_time_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
    
    # NEW: Try to identify date columns that are still objects
    if len(date_time_cols) == 0:
        for col in df.select_dtypes(include=['object']).columns:
            # Check if column name contains date-like words or content looks like date
            col_lower = col.lower()
            if any(word in col_lower for word in ['date', 'time', 'timestamp', 'year', 'month']):
                try:
                    # Sample some values to check if they are parseable
                    sample_vals = df[col].dropna().head(100)
                    if len(sample_vals) > 0:
                        pd.to_datetime(sample_vals, errors='raise')
                        return "Time_series"
                except:
                    continue

    if len(date_time_cols) > 0:
        return "Time_series"
    
    if unique == 2 and (target_column_dtype == object or target_column_dtype == bool or target_column_dtype.name == 'category'):
        return "Binary_classification"
    elif unique > 2 and (target_column_dtype == object or target_column_dtype.name == 'category'):
        return "Multi_classification"
    elif (target_column_dtype == int or target_column_dtype == float or np.issubdtype(target_column_dtype, np.number)):
        return "Regression"
    else:
        return "Classification" # Fallback

