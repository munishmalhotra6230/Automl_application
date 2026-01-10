"""
Quick Test Script for Advanced Features
Tests that all new modules are working correctly
"""

print("🧪 Testing Advanced Features...")
print("="*60)

# Test 1: Hyperparameter Tuning
print("\n1️⃣ Testing Hyperparameter Optimization...")
try:
    from hyper_parameter_tuning.advanced_tuning import AdvancedHyperparameterTuner
    print("   ✅ Hyperparameter tuning module loaded successfully!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Model Explainability
print("\n2️⃣ Testing Model Explainability...")
try:
    from model_explainability.explainer import ModelExplainer
    print("   ✅ Model explainability module loaded successfully!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Ensemble Methods
print("\n3️⃣ Testing Ensemble Methods...")
try:
    from ensemble.ensemble_methods import StackingEnsemble, BlendingEnsemble
    print("   ✅ Ensemble methods module loaded successfully!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Data Validation
print("\n4️⃣ Testing Data Validation...")
try:
    from data_validation.validator import DataValidator
    print("   ✅ Data validation module loaded successfully!")
    
    # Quick validation test
    import pandas as pd
    import numpy as np
    
    # Create sample data
    test_df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B', 'C'], 100),
        'feature3': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Add some missing values
    test_df.loc[0:5, 'feature1'] = np.nan
    
    validator = DataValidator(test_df, 'target')
    report = validator.run_full_validation()
    
    print(f"   📊 Quality Score: {report['quality_score']:.1f}/100")
    print(f"   📈 Dataset: {report['basic_info']['num_rows']} rows, {report['basic_info']['num_columns']} columns")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Monitoring
print("\n5️⃣ Testing Model Monitoring...")
try:
    from monitoring.model_monitor import ModelMonitor
    print("   ✅ Monitoring module loaded successfully!")
    
    # Quick monitoring test
    monitor = ModelMonitor("test_model")
    monitor.log_prediction(
        input_data={'test': 1},
        prediction=0.85,
        latency_ms=12.3
    )
    
    metrics = monitor.get_performance_metrics()
    if 'error' not in metrics:
        print(f"   ⚡ Average latency: {metrics['latency']['avg_ms']:.2f}ms")
        print(f"   📊 Total predictions logged: {metrics['total_predictions']}")
    
    # Clean up test logs
    monitor.clear_logs()
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🎉 All modules tested!")
print("="*60)

print("\n📚 Next Steps:")
print("   1. Check ADVANCED_FEATURES_README.md for full documentation")
print("   2. Install dependencies: pip install -r requirements.txt")
print("   3. Integrate endpoints from ADVANCED_FEATURES_INTEGRATION.py")
print("   4. Start using the new features in your AutoML pipeline!")
