"""
Model Performance Monitoring Module
Track model performance, latency, and provide dashboards
"""
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class ModelMonitor:
    """
    Monitor model performance over time
    """
    
    def __init__(self, model_id, storage_path="monitoring_logs"):
        """
        Args:
            model_id: Unique identifier for the model
            storage_path: Directory to store monitoring logs
        """
        self.model_id = model_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.log_file = self.storage_path / f"{model_id}_monitor.jsonl"
        self.stats = {
            "total_predictions": 0,
            "total_errors": 0,
            "avg_latency_ms": 0,
            "predictions_by_day": {},
        }
        
        self._load_existing_logs()
    
    def _load_existing_logs(self):
        """Load existing monitoring logs if available"""
        if self.log_file.exists():
            try:
                logs = []
                with open(self.log_file, 'r') as f:
                    for line in f:
                        logs.append(json.loads(line))
                
                if logs:
                    self.stats["total_predictions"] = len(logs)
                    latencies = [log.get("latency_ms", 0) for log in logs]
                    self.stats["avg_latency_ms"] = np.mean(latencies)
                    
                    print(f"📊 Loaded {len(logs)} previous monitoring logs for {self.model_id}")
            except:
                pass
    
    def log_prediction(self, input_data, prediction, latency_ms, 
                      actual_value=None, metadata=None):
        """
        Log a single prediction
        
        Args:
            input_data: Input features (dict or array)
            prediction: Model prediction
            latency_ms: Prediction latency in milliseconds
            actual_value: Actual value (if available)
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_id": self.model_id,
            "prediction": float(prediction) if not isinstance(prediction, (list, dict)) else prediction,
            "latency_ms": float(latency_ms),
            "actual_value": float(actual_value) if actual_value is not None else None,
            "metadata": metadata or {}
        }
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Update stats
        self.stats["total_predictions"] += 1
        
        # Update average latency (rolling average)
        n = self.stats["total_predictions"]
        self.stats["avg_latency_ms"] = (
            (self.stats["avg_latency_ms"] * (n-1) + latency_ms) / n
        )
    
    def log_error(self, error_msg, input_data=None):
        """Log prediction error"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_id": self.model_id,
            "error": True,
            "error_message": str(error_msg),
            "input_data": str(input_data) if input_data is not None else None
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')
        
        self.stats["total_errors"] += 1
    
    def get_performance_metrics(self, last_n_predictions=None):
        """
        Calculate performance metrics from logs
        
        Args:
            last_n_predictions: Analyze only last N predictions
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.log_file.exists():
            return {"error": "No logs available"}
        
        # Read logs
        logs = []
        with open(self.log_file, 'r') as f:
            for line in f:
                logs.append(json.loads(line))
        
        if not logs:
            return {"error": "No logs available"}
        
        # Filter to last N if specified
        if last_n_predictions:
            logs = logs[-last_n_predictions:]
        
        # Filter out errors
        prediction_logs = [log for log in logs if not log.get("error", False)]
        error_logs = [log for log in logs if log.get("error", False)]
        
        if not prediction_logs:
            return {"error": "No prediction logs available"}
        
        # Calculate metrics
        latencies = [log["latency_ms"] for log in prediction_logs]
        
        metrics = {
            "total_predictions": len(prediction_logs),
            "total_errors": len(error_logs),
            "error_rate": len(error_logs) / len(logs) if logs else 0,
            "latency": {
                "avg_ms": np.mean(latencies),
                "median_ms": np.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "min_ms": np.min(latencies),
                "max_ms": np.max(latencies),
            },
        }
        
        # Calculate accuracy if actual values available
        logs_with_actuals = [
            log for log in prediction_logs 
            if log.get("actual_value") is not None
        ]
        
        if logs_with_actuals:
            predictions = np.array([log["prediction"] for log in logs_with_actuals])
            actuals = np.array([log["actual_value"] for log in logs_with_actuals])
            
            # Check if classification or regression
            is_classification = len(np.unique(actuals)) < 20
            
            if is_classification:
                accuracy = (predictions == actuals).mean()
                metrics["accuracy"] = float(accuracy)
            else:
                mse = np.mean((predictions - actuals) ** 2)
                mae = np.mean(np.abs(predictions - actuals))
                metrics["mse"] = float(mse)
                metrics["mae"] = float(mae)
                metrics["rmse"] = float(np.sqrt(mse))
        
        return metrics
    
    def get_dashboard_data(self):
        """
        Get data for monitoring dashboard
        
        Returns:
            Dictionary with dashboard data
        """
        metrics = self.get_performance_metrics()
        
        if "error" in metrics:
            return metrics
        
        # Read recent logs for trend analysis
        recent_logs = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                all_logs = [json.loads(line) for line in f]
                recent_logs = all_logs[-100:]  # Last 100 predictions
        
        # Group by date
        predictions_by_date = {}
        latency_by_date = {}
        
        for log in recent_logs:
            if log.get("error"):
                continue
            
            date = log["timestamp"][:10]  # YYYY-MM-DD
            
            predictions_by_date[date] = predictions_by_date.get(date, 0) + 1
            
            if date not in latency_by_date:
                latency_by_date[date] = []
            latency_by_date[date].append(log["latency_ms"])
        
        # Calculate average latency per date
        avg_latency_by_date = {
            date: np.mean(latencies) 
            for date, latencies in latency_by_date.items()
        }
        
        dashboard = {
            "model_id": self.model_id,
            "summary": metrics,
            "trends": {
                "predictions_per_day": predictions_by_date,
                "avg_latency_per_day": avg_latency_by_date,
            },
            "health_status": self._get_health_status(metrics),
        }
        
        return dashboard
    
    def _get_health_status(self, metrics):
        """Determine model health status"""
        
        # Check latency
        if metrics["latency"]["avg_ms"] > 1000:
            latency_status = "⚠️  SLOW"
        elif metrics["latency"]["avg_ms"] > 500:
            latency_status = "⚡ MODERATE"
        else:
            latency_status = "✅ FAST"
        
        # Check error rate
        if metrics["error_rate"] > 0.1:
            error_status = "🔴 HIGH ERRORS"
        elif metrics["error_rate"] > 0.01:
            error_status = "🟡 SOME ERRORS"
        else:
            error_status = "✅ LOW ERRORS"
        
        # Overall health
        if metrics["error_rate"] > 0.1 or metrics["latency"]["avg_ms"] > 1000:
            overall = "🔴 UNHEALTHY"
        elif metrics["error_rate"] > 0.01 or metrics["latency"]["avg_ms"] > 500:
            overall = "🟡 DEGRADED"
        else:
            overall = "✅ HEALTHY"
        
        return {
            "overall": overall,
            "latency": latency_status,
            "errors": error_status,
        }
    
    def clear_logs(self):
        """Clear all monitoring logs"""
        if self.log_file.exists():
            self.log_file.unlink()
        
        self.stats = {
            "total_predictions": 0,
            "total_errors": 0,
            "avg_latency_ms": 0,
            "predictions_by_day": {},
        }
        
        print(f"🗑️  Cleared monitoring logs for {self.model_id}")


def monitor_prediction(model, input_data, model_id="default", actual_value=None):
    """
    Convenience function to monitor a single prediction
    
    Args:
        model: Trained model
        input_data: Input features
        model_id: Model identifier
        actual_value: Actual value (optional)
    
    Returns:
        Dictionary with prediction and monitoring info
    """
    monitor = ModelMonitor(model_id)
    
    # Time the prediction
    start_time = time.time()
    
    try:
        prediction = model.predict(input_data)
        latency_ms = (time.time() - start_time) * 1000
        
        # Log the prediction
        monitor.log_prediction(
            input_data=input_data,
            prediction=prediction,
            latency_ms=latency_ms,
            actual_value=actual_value
        )
        
        return {
            "prediction": prediction,
            "latency_ms": latency_ms,
            "status": "success"
        }
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        monitor.log_error(str(e), input_data)
        
        return {
            "error": str(e),
            "latency_ms": latency_ms,
            "status": "error"
        }


def get_all_model_health():
    """
    Get health status of all monitored models
    
    Returns:
        Dictionary with health status of all models
    """
    storage_path = Path("monitoring_logs")
    
    if not storage_path.exists():
        return {"models": []}
    
    all_models = []
    
    for log_file in storage_path.glob("*_monitor.jsonl"):
        model_id = log_file.stem.replace("_monitor", "")
        monitor = ModelMonitor(model_id)
        
        try:
            dashboard = monitor.get_dashboard_data()
            all_models.append({
                "model_id": model_id,
                "health": dashboard.get("health_status", {}),
                "total_predictions": dashboard.get("summary", {}).get("total_predictions", 0),
                "avg_latency_ms": dashboard.get("summary", {}).get("latency", {}).get("avg_ms", 0),
            })
        except:
            pass
    
    return {"models": all_models}
