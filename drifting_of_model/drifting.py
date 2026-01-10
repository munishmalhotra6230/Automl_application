from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, mean_squared_error
import datetime
import time
def monitoring_alert_system(is_drifted, performance_score, min_threshold):
    if is_drifted or performance_score < min_threshold:
        print("🚨 CRITICAL: Triggering Automated Retraining Pipeline...")
        # Yahan aapka purana Model_zoo + trainer_engine call hoga
        return True # Retrain Triggered
    return False


def monitor_latency(start_time):
    """
    API call ke start aur end ka difference nikalna.
    """
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    if latency_ms > 200: # 200ms threshold for 2026
        print(f"🐢 Slow Prediction: {latency_ms:.2f}ms")
    
    return latency_ms

# FastAPI integration example:
# start = time.time()
# prediction = model.predict(df)
# latency = monitor_latency(start)

def monitor_performance(y_actual, y_pred, problem_type):
    # Metrics fetch karna (Pehle banaye huye evaluation logic se)
    if problem_type == "classification":
        score = accuracy_score(y_actual, y_pred)
    else:
        score = mean_squared_error(y_actual, y_pred)
    
    # Save performance to a log file for dashboarding
    with open("performance_logs.csv", "a") as f:
        f.write(f"{datetime.now()},{score}\n")
    return score


def check_data_drift(train_df, production_df, threshold=0.05):
    """
    Check if production data distribution has drifted from training data.
    """
    drift_report = {}
    is_drifted = False

    numeric_cols = train_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Kolmogorov-Smirnov test (2026 Standard)
        stat, p_value = ks_2samp(train_df[col], production_df[col])
        
        # Agar p-value threshold se kam hai, toh distribution badal gayi hai
        drift_report[col] = {"p_value": p_value, "drift_detected": p_value < threshold}
        if p_value < threshold:
            is_drifted = True
            print(f"⚠️ DRIFT DETECTED in column: {col}")
            
    return is_drifted, drift_report
