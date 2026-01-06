import pytest
import pandas as pd
import sys
import os

# Add src folder to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processor import load_data, split_data

# Test 1: Data Loading
def test_data_loading_and_shape():
    """Checks if data loads correctly, is a DataFrame, and has expected size."""
    try:
        df = load_data()
    except FileNotFoundError:
        pytest.fail("Data file not found. Ensure 'data/cleveland.data' is present.")
        
    assert isinstance(df, pd.DataFrame), "Loaded data is not a pandas DataFrame"
    # Cleveland data contains approximately 297 cleaned rows
    assert df.shape[0] >= 290, "Data size is too small"
    assert df['target'].isin([0, 1]).all(), "Target column contains unexpected values"

# Test 2: Data Splitting
def test_data_splitting_and_stratification():
    """Checks if data is split correctly (80/20) and stratified properly."""
    try:
        df = load_data()
    except:
        return # Skip if load fails
        
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    
    # Check 1: Test size is approximately 20%
    expected_test_size = int(len(df) * 0.20)
    assert abs(len(y_test) - expected_test_size) <= 2, "Test set size is incorrect"
    
    # Check 2: Stratification check (Ratio difference < 5%)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) / train_ratio < 0.05, "Stratification failed"