"""
Advanced Ensemble Methods Module
Implements Stacking, Blending, and Weighted Averaging
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class StackingEnsemble:
    """
    Stacking ensemble that uses predictions from base models as features for a meta-model
    """
    
    def __init__(self, base_models, meta_model=None, problem_type='Classification', cv=5):
        """
        Args:
            base_models: List of (name, model) tuples
            meta_model: Meta-learner model (auto-selected if None)
            problem_type: 'Classification' or 'Regression'
            cv: Number of CV folds
        """
        self.base_models = base_models
        self.problem_type = problem_type
        self.cv = cv
        self.fitted_base_models = []
        
        # Auto-select meta-model if not provided
        if meta_model is None:
            if problem_type == 'Classification':
                self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                self.meta_model = Ridge(alpha=1.0, random_state=42)
        else:
            self.meta_model = meta_model
        
        print(f"🎯 Stacking Ensemble initialized with {len(base_models)} base models")
    
    def fit(self, X_train, y_train):
        """
        Fit stacking ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"📚 Training {len(self.base_models)} base models with {self.cv}-fold CV...")
        
        # Generate out-of-fold predictions for meta-features
        meta_features = np.zeros((len(X_train), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models):
            print(f"  ├─ Training {name}...")
            
            # Get out-of-fold predictions using cross-validation
            try:
                oof_preds = cross_val_predict(
                    clone(model), X_train, y_train, 
                    cv=self.cv, 
                    n_jobs=-1,
                    method='predict'
                )
                meta_features[:, i] = oof_preds
                
                # Fit on full training data
                fitted_model = clone(model)
                fitted_model.fit(X_train, y_train)
                self.fitted_base_models.append((name, fitted_model))
                
            except Exception as e:
                print(f"    ⚠️  {name} failed: {e}")
                meta_features[:, i] = 0  # Fill with zeros if failed
                self.fitted_base_models.append((name, None))
        
        # Train meta-model on meta-features
        print(f"  └─ Training meta-model...")
        self.meta_model.fit(meta_features, y_train)
        
        print(f"✅ Stacking ensemble training complete!")
        
        return self
    
    def predict(self, X_test):
        """
        Make predictions using stacking ensemble
        
        Args:
            X_test: Test features
        
        Returns:
            Predictions
        """
        # Get predictions from base models
        meta_features = np.zeros((len(X_test), len(self.fitted_base_models)))
        
        for i, (name, model) in enumerate(self.fitted_base_models):
            if model is not None:
                try:
                    meta_features[:, i] = model.predict(X_test)
                except:
                    meta_features[:, i] = 0
        
        # Meta-model predicts on base model predictions
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X_test):
        """Predict probabilities (for classification only)"""
        if self.problem_type != 'Classification':
            raise ValueError("predict_proba only available for classification")
        
        # Get predictions from base models
        meta_features = np.zeros((len(X_test), len(self.fitted_base_models)))
        
        for i, (name, model) in enumerate(self.fitted_base_models):
            if model is not None and hasattr(model, 'predict_proba'):
                try:
                    # Use probability of positive class
                    proba = model.predict_proba(X_test)
                    meta_features[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
                except:
                    meta_features[:, i] = 0.5
            else:
                meta_features[:, i] = 0.5
        
        # Meta-model predicts probabilities
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            # Fallback to predictions converted to probabilities
            preds = self.meta_model.predict(meta_features)
            return np.column_stack([1-preds, preds])


class BlendingEnsemble:
    """
    Blending ensemble using weighted average of predictions
    """
    
    def __init__(self, models, weights=None, problem_type='Classification'):
        """
        Args:
            models: List of (name, fitted_model) tuples
            weights: List of weights (auto-calculated if None)
            problem_type: 'Classification' or 'Regression'
        """
        self.models = models
        self.problem_type = problem_type
        
        # Auto-assign equal weights if not provided
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()  # Normalize
        
        print(f"🎨 Blending Ensemble with {len(models)} models")
        print(f"   Weights: {dict(zip([name for name, _ in models], self.weights))}")
    
    def predict(self, X_test):
        """
        Make weighted average predictions
        
        Args:
            X_test: Test features
        
        Returns:
            Blended predictions
        """
        predictions = np.zeros((len(X_test), len(self.models)))
        
        for i, (name, model) in enumerate(self.models):
            try:
                predictions[:, i] = model.predict(X_test)
            except Exception as e:
                print(f"⚠️  {name} prediction failed: {e}")
                predictions[:, i] = 0
        
        # Weighted average
        blended = np.average(predictions, axis=1, weights=self.weights)
        
        return blended
    
    def predict_proba(self, X_test):
        """Predict probabilities (classification only)"""
        if self.problem_type != 'Classification':
            raise ValueError("predict_proba only for classification")
        
        # Get probability predictions
        all_probas = []
        
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                try:
                    all_probas.append(model.predict_proba(X_test))
                except:
                    pass
        
        if not all_probas:
            raise ValueError("No models support predict_proba")
        
        # Weighted average of probabilities
        weighted_proba = np.average(all_probas, axis=0, weights=self.weights[:len(all_probas)])
        
        return weighted_proba


def create_optimal_weights(models, X_val, y_val, problem_type='Classification'):
    """
    Calculate optimal weights for blending based on validation performance
    
    Args:
        models: List of (name, model) tuples
        X_val: Validation features
        y_val: Validation labels
        problem_type: Problem type
    
    Returns:
        Array of optimal weights
    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    scores = []
    
    for name, model in models:
        try:
            preds = model.predict(X_val)
            
            if problem_type == 'Classification':
                score = accuracy_score(y_val, preds)
            else:
                score = -mean_squared_error(y_val, preds)  # Negative MSE
            
            scores.append(max(score, 0))  # Ensure non-negative
        except:
            scores.append(0)
    
    # Convert scores to weights (higher score = higher weight)
    scores = np.array(scores)
    weights = scores / scores.sum() if scores.sum() > 0 else np.ones(len(scores)) / len(scores)
    
    print(f"📊 Calculated optimal weights based on validation performance:")
    for (name, _), weight in zip(models, weights):
        print(f"   {name:30s}: {weight:.4f}")
    
    return weights


def create_voting_ensemble(models, X_val, y_val, problem_type='Classification', 
                           method='weighted'):
    """
    Create ensemble using voting or averaging
    
    Args:
        models: List of (name, model) tuples
        X_val: Validation data for weight calculation
        y_val: Validation labels
        problem_type: Problem type
        method: 'weighted' or 'equal'
    
    Returns:
        BlendingEnsemble instance
    """
    if method == 'weighted':
        weights = create_optimal_weights(models, X_val, y_val, problem_type)
    else:
        weights = None  # Equal weights
    
    return BlendingEnsemble(models, weights, problem_type)


def auto_ensemble(fitted_models, X_train, y_train, X_val, y_val, 
                  problem_type='Classification', top_n=5, method='stacking'):
    """
    Automatically create best ensemble from fitted models
    
    Args:
        fitted_models: Dictionary of {name: model}
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        problem_type: Problem type
        top_n: Number of top models to include
        method: 'stacking' or 'blending'
    
    Returns:
        Ensemble model
    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    # Evaluate all models
    model_scores = {}
    
    for name, model in fitted_models.items():
        try:
            preds = model.predict(X_val)
            
            if problem_type == 'Classification':
                score = accuracy_score(y_val, preds)
            else:
                score = -mean_squared_error(y_val, preds)
            
            model_scores[name] = score
        except:
            model_scores[name] = -np.inf
    
    # Get top N models
    top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_model_list = [(name, fitted_models[name]) for name, score in top_models]
    
    print(f"\n🏆 Top {len(top_model_list)} models selected for ensemble:")
    for name, score in top_models:
        print(f"   {name:30s}: {score:.4f}")
    
    # Create ensemble
    if method == 'stacking':
        ensemble = StackingEnsemble(
            [(name, clone(model)) for name, model in top_model_list],
            problem_type=problem_type
        )
        ensemble.fit(X_train, y_train)
    else:  # blending
        weights = create_optimal_weights(top_model_list, X_val, y_val, problem_type)
        ensemble = BlendingEnsemble(top_model_list, weights, problem_type)
    
    return ensemble
