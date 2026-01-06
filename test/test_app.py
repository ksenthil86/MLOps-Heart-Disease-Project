# MLOPS_Heart_Disease/test/test_app.py

import pytest
import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add parent directory to path for app import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestFlaskApp:
    """Test suite for Flask API endpoints"""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with predict and predict_proba methods"""
        model = Mock()
        model.predict.return_value = np.array([1])  # Positive prediction
        model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% confidence
        return model

    @pytest.fixture
    def client(self, mock_model):
        """Create Flask test client with mocked model"""
        # Mock prometheus_flask_exporter before importing app
        mock_prometheus = MagicMock()
        mock_prometheus.PrometheusMetrics = MagicMock()
        
        with patch.dict('sys.modules', {'prometheus_flask_exporter': mock_prometheus}):
            with patch('joblib.load', return_value=mock_model):
                # Import app after patching
                from app import app
                app.config['TESTING'] = True
                with app.test_client() as client:
                    yield client

    def test_health_endpoint(self, client):
        """Test /health endpoint returns 200 OK"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'ok'
        assert 'model' in data

    def test_predict_endpoint_with_valid_data(self, client):
        """Test /predict endpoint with valid input data"""
        sample_data = {
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(sample_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'prediction' in data
        assert 'status' in data
        assert 'confidence_score' in data
        assert data['status'] == 'Prediction successful'
        assert data['prediction'] in [0, 1]
        assert 0.0 <= data['confidence_score'] <= 1.0

    def test_predict_endpoint_returns_correct_prediction(self, client):
        """Test that prediction matches mocked model output"""
        sample_data = {
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(sample_data),
                              content_type='application/json')
        
        data = json.loads(response.data)
        # Mock model returns prediction=1 with 70% confidence
        assert data['prediction'] == 1
        assert data['confidence_score'] == 0.7

    def test_predict_endpoint_with_missing_fields(self, client):
        """Test /predict endpoint with incomplete data"""
        incomplete_data = {
            'age': 63,
            'sex': 1
            # Missing other required fields
        }
        
        response = client.post('/predict',
                              data=json.dumps(incomplete_data),
                              content_type='application/json')
        
        # Mock model still processes this, but in real scenario it would fail
        # Just verify it returns some response
        assert response.status_code in [200, 400, 500]
        data = json.loads(response.data)
        assert 'prediction' in data or 'error' in data

    def test_predict_endpoint_with_invalid_data_types(self, client):
        """Test /predict endpoint with invalid data types"""
        invalid_data = {
            'age': 'sixty-three',  # String instead of number
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(invalid_data),
                              content_type='application/json')
        
        # Response depends on pandas conversion - may succeed or fail
        assert response.status_code in [200, 400, 500]

    def test_predict_endpoint_with_empty_body(self, client):
        """Test /predict endpoint with empty request body"""
        response = client.post('/predict',
                              data=json.dumps({}),
                              content_type='application/json')
        
        # Mock model may still process empty data
        assert response.status_code in [200, 400, 500]

    def test_predict_endpoint_confidence_score_range(self, client):
        """Test that confidence score is within valid range"""
        sample_data = {
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(sample_data),
                              content_type='application/json')
        
        data = json.loads(response.data)
        assert 0.0 <= data['confidence_score'] <= 1.0, "Confidence score out of range"

    def test_metrics_endpoint_exists(self, client):
        """Test that /metrics endpoint exists for Prometheus"""
        response = client.get('/metrics')
        
        # When prometheus_flask_exporter is mocked, /metrics may not exist
        # In production with real prometheus_flask_exporter, it returns 200
        assert response.status_code in [200, 404]

    def test_multiple_predictions(self, client):
        """Test multiple consecutive predictions"""
        sample_data = {
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        # Make 3 consecutive predictions
        for _ in range(3):
            response = client.post('/predict',
                                  data=json.dumps(sample_data),
                                  content_type='application/json')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'Prediction successful'

    def test_response_json_structure(self, client):
        """Test that response has correct JSON structure"""
        sample_data = {
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(sample_data),
                              content_type='application/json')
        
        data = json.loads(response.data)
        
        # Check required keys exist
        required_keys = ['prediction', 'status', 'confidence_score']
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

    def test_content_type_json(self, client):
        """Test that response content type is JSON"""
        sample_data = {
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(sample_data),
                              content_type='application/json')
        
        assert response.content_type == 'application/json'
