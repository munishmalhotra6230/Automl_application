"""
Model Explainability Module using SHAP
Provides interpretability for ML predictions
"""
import shap
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """
    Provides model explanations using SHAP values
    """
    
    def __init__(self, model, X_train, problem_type='Classification'):
        """
        Initialize explainer
        
        Args:
            model: Trained model
            X_train: Training data (used for background)
            problem_type: Type of ML problem
        """
        self.model = model
        self.X_train = X_train
        self.problem_type = problem_type
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Create appropriate SHAP explainer based on model type"""
        try:
            model_type = type(self.model).__name__.lower()
            
            # Sample background data if too large
            if len(self.X_train) > 100:
                background = shap.sample(self.X_train, 100)
            else:
                background = self.X_train
            
            # Choose explainer type
            if 'xgb' in model_type or 'lightgbm' in model_type or 'catboost' in model_type:
                # Tree-based explainer (fast and exact)
                self.explainer = shap.TreeExplainer(self.model)
            elif 'random' in model_type or 'forest' in model_type:
                # Tree explainer for forests
                self.explainer = shap.TreeExplainer(self.model)
            elif 'linear' in model_type or 'logistic' in model_type or 'ridge' in model_type:
                # Linear explainer
                self.explainer = shap.LinearExplainer(self.model, background)
            else:
                # Kernel explainer (model-agnostic, slower)
                self.explainer = shap.KernelExplainer(self.model.predict, background)
                
            print(f"✅ SHAP Explainer initialized: {type(self.explainer).__name__}")
            
        except Exception as e:
            print(f"⚠️  Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, X_single, return_plot_data=False):
        """
        Explain a single prediction
        
        Args:
            X_single: Single sample to explain (1D array or Series)
            return_plot_data: If True, return data for plotting
        
        Returns:
            Dictionary with explanation data
        """
        if self.explainer is None:
            return {"error": "Explainer not available"}
        
        try:
            # Ensure X_single is 2D
            if isinstance(X_single, pd.Series):
                X_single = X_single.to_frame().T
            elif len(X_single.shape) == 1:
                X_single = X_single.reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(X_single)
            
            # Handle multi-class classification
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for now
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Get first sample
            
            # Get feature names
            if hasattr(X_single, 'columns'):
                feature_names = X_single.columns.tolist()
            else:
                feature_names = [f"Feature_{i}" for i in range(len(shap_values))]
            
            # Create feature importance dictionary
            feature_importance = {
                name: float(value) 
                for name, value in zip(feature_names, shap_values)
            }
            
            # Sort by absolute importance
            sorted_importance = dict(
                sorted(feature_importance.items(), 
                       key=lambda x: abs(x[1]), 
                       reverse=True)
            )
            
            # Get base value (expected value)
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            else:
                base_value = float(base_value)
            
            # Make prediction
            prediction = self.model.predict(X_single)[0]
            
            result = {
                "prediction": float(prediction),
                "base_value": base_value,
                "feature_contributions": sorted_importance,
                "top_positive": self._get_top_features(sorted_importance, positive=True),
                "top_negative": self._get_top_features(sorted_importance, positive=False),
            }
            
            if return_plot_data:
                result['shap_values'] = shap_values.tolist()
                result['feature_values'] = X_single.values[0].tolist() if hasattr(X_single, 'values') else list(X_single[0])
            
            return result
            
        except Exception as e:
            return {"error": f"Explanation failed: {str(e)}"}
    
    def _get_top_features(self, importance_dict, positive=True, top_n=5):
        """Get top contributing features"""
        if positive:
            items = [(k, v) for k, v in importance_dict.items() if v > 0]
        else:
            items = [(k, v) for k, v in importance_dict.items() if v < 0]
        
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        return dict(items[:top_n])
    
    def get_global_feature_importance(self, X_sample=None, max_samples=100):
        """
        Get global feature importance across dataset
        
        Args:
            X_sample: Sample of data to analyze (uses X_train if None)
            max_samples: Maximum samples to use for analysis
        
        Returns:
            Dictionary with global importance values
        """
        if self.explainer is None:
            return {"error": "Explainer not available"}
        
        try:
            # Use sample of training data
            if X_sample is None:
                X_sample = self.X_train
            
            # Limit samples for speed
            if len(X_sample) > max_samples:
                indices = np.random.choice(len(X_sample), max_samples, replace=False)
                X_sample = X_sample.iloc[indices] if hasattr(X_sample, 'iloc') else X_sample[indices]
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-class
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get feature names
            if hasattr(X_sample, 'columns'):
                feature_names = X_sample.columns.tolist()
            else:
                feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]
            
            # Calculate mean absolute SHAP value for each feature
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Create importance dictionary
            importance = {
                name: float(value)
                for name, value in zip(feature_names, mean_shap)
            }
            
            # Sort by importance
            sorted_importance = dict(
                sorted(importance.items(), 
                       key=lambda x: x[1], 
                       reverse=True)
            )
            
            return {
                "global_importance": sorted_importance,
                "top_features": dict(list(sorted_importance.items())[:10]),
                "total_samples_analyzed": len(X_sample)
            }
            
        except Exception as e:
            return {"error": f"Global importance calculation failed: {str(e)}"}
    
    def explain_batch(self, X_batch, max_samples=50):
        """
        Explain multiple predictions
        
        Args:
            X_batch: Multiple samples
            max_samples: Max number to explain
        
        Returns:
            List of explanations
        """
        if len(X_batch) > max_samples:
            indices = np.random.choice(len(X_batch), max_samples, replace=False)
            X_batch = X_batch.iloc[indices] if hasattr(X_batch, 'iloc') else X_batch[indices]
        
        explanations = []
        for i in range(len(X_batch)):
            sample = X_batch.iloc[i] if hasattr(X_batch, 'iloc') else X_batch[i]
            exp = self.explain_prediction(sample)
            explanations.append(exp)
        
        return explanations


def create_explainer(model, X_train, problem_type='Classification'):
    """
    Convenience function to create model explainer
    
    Args:
        model: Trained model
        X_train: Training data
        problem_type: Problem type
    
    Returns:
        ModelExplainer instance
    """
    return ModelExplainer(model, X_train, problem_type)


def explain_predictions(model, X_train, X_predict, problem_type='Classification', 
                        global_importance=True):
    """
    Get both local and global explanations
    
    Args:
        model: Trained model
        X_train: Training data (for background)
        X_predict: Data to explain
        problem_type: Problem type
        global_importance: Whether to include global importance
    
    Returns:
        Dictionary with explanations
    """
    explainer = ModelExplainer(model, X_train, problem_type)
    
    results = {
        "local_explanations": []
    }
    
    # Get local explanations
    if len(X_predict) == 1:
        # Single prediction
        results["local_explanations"] = [explainer.explain_prediction(X_predict)]
    else:
        # Batch predictions (limit to avoid slowness)
        results["local_explanations"] = explainer.explain_batch(X_predict, max_samples=20)
    
    # Get global importance
    if global_importance:
        results["global_feature_importance"] = explainer.get_global_feature_importance()
    
    return results
