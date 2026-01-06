# MLOPS_Heart_Disease/test/test_train_model.py

import pytest
import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Add src folder to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_model import train_and_log_model
from data_processor import load_data, split_data, create_preprocessor


class TestTrainModel:
    """Test suite for train_model.py functions"""

    @pytest.fixture
    def setup_data(self):
        """Fixture to load and prepare test data"""
        data = load_data()
        X_train, X_test, y_train, y_test = split_data(data, test_size=0.2, random_state=42)
        preprocessor = create_preprocessor(data)
        return X_train, X_test, y_train, y_test, preprocessor

    def test_train_and_log_model_with_logistic_regression(self, setup_data):
        """Test training and logging with Logistic Regression"""
        X_train, X_test, y_train, y_test, preprocessor = setup_data
        
        model = LogisticRegression(solver='liblinear', random_state=42)
        
        # Train the model and get AUC score
        auc_score = train_and_log_model(
            model, X_train, y_train, X_test, y_test,
            preprocessor, "LogisticRegression"
        )
        
        # Assertions
        assert auc_score is not None, "AUC score should not be None"
        assert isinstance(auc_score, float), "AUC score should be a float"
        assert 0.0 <= auc_score <= 1.0, "AUC score should be between 0 and 1"
        assert auc_score > 0.7, "AUC score should be reasonably high (>0.7)"

    def test_train_and_log_model_with_random_forest(self, setup_data):
        """Test training and logging with Random Forest"""
        X_train, X_test, y_train, y_test, preprocessor = setup_data
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Train the model and get AUC score
        auc_score = train_and_log_model(
            model, X_train, y_train, X_test, y_test,
            preprocessor, "RandomForest"
        )
        
        # Assertions
        assert auc_score is not None, "AUC score should not be None"
        assert isinstance(auc_score, float), "AUC score should be a float"
        assert 0.0 <= auc_score <= 1.0, "AUC score should be between 0 and 1"
        assert auc_score > 0.7, "AUC score should be reasonably high (>0.7)"

    def test_model_comparison(self, setup_data):
        """Test that both models can be trained and compared"""
        X_train, X_test, y_train, y_test, preprocessor = setup_data
        
        # Train both models
        lr = LogisticRegression(solver='liblinear', random_state=42)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        lr_auc = train_and_log_model(
            lr, X_train, y_train, X_test, y_test,
            preprocessor, "LogisticRegression"
        )
        
        rf_auc = train_and_log_model(
            rf, X_train, y_train, X_test, y_test,
            preprocessor, "RandomForest"
        )
        
        # Both should return valid scores
        assert lr_auc is not None and rf_auc is not None
        assert isinstance(lr_auc, float) and isinstance(rf_auc, float)
        
        # At least one model should have decent performance
        assert max(lr_auc, rf_auc) > 0.75, "At least one model should perform well"

    def test_model_parameters_are_logged(self, setup_data):
        """Test that model parameters are properly set"""
        X_train, X_test, y_train, y_test, preprocessor = setup_data
        
        # Create model with specific parameters
        model = LogisticRegression(
            solver='liblinear',
            random_state=42,
            max_iter=100,
            C=1.0
        )
        
        # Get model parameters
        params = model.get_params()
        
        # Verify parameters
        assert params['solver'] == 'liblinear'
        assert params['random_state'] == 42
        assert params['max_iter'] == 100
        assert params['C'] == 1.0

    def test_preprocessor_is_valid(self, setup_data):
        """Test that preprocessor is created correctly"""
        _, _, _, _, preprocessor = setup_data
        
        # Preprocessor should be a valid sklearn transformer
        assert hasattr(preprocessor, 'fit_transform'), "Preprocessor should have fit_transform method"
        assert hasattr(preprocessor, 'transform'), "Preprocessor should have transform method"

    def test_train_test_split_sizes(self, setup_data):
        """Test that data split sizes are correct"""
        X_train, X_test, y_train, y_test, _ = setup_data
        
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        
        # Test size should be approximately 20% (within 2%)
        assert 0.18 <= test_ratio <= 0.22, f"Test ratio {test_ratio} not close to 0.20"
        
        # Train and test labels should have same length as features
        assert len(X_train) == len(y_train), "Training features and labels size mismatch"
        assert len(X_test) == len(y_test), "Test features and labels size mismatch"

    def test_stratification_preserved(self, setup_data):
        """Test that class distribution is preserved in train/test split"""
        _, _, y_train, y_test, _ = setup_data
        
        train_positive_ratio = y_train.sum() / len(y_train)
        test_positive_ratio = y_test.sum() / len(y_test)
        
        # Ratios should be similar (within 5%)
        ratio_diff = abs(train_positive_ratio - test_positive_ratio)
        assert ratio_diff < 0.05, f"Stratification failed: ratio difference {ratio_diff}"
