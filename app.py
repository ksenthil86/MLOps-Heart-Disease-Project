import joblib
from flask import Flask, request, jsonify
import pandas as pd
import logging
from datetime import datetime
from prometheus_flask_exporter import PrometheusMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 1. Model Loading
# Load the trained model from pickle file
MODEL_PATH = "/app/models/heartDisease_Classifier.pkl"
MODEL_NAME = "RandomForest_Heart_Classifier"

try:
    # Load model from pickle file using joblib
    loaded_model = joblib.load(MODEL_PATH)
    logger.info(f"✓ Model '{MODEL_NAME}' successfully loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"✗ Failed to load model from {MODEL_PATH}: {str(e)}")
    # If model fails to load, fail the application
    raise

# 2. Flask Application Setup
app = Flask(__name__)

# 3. Setup Prometheus metrics
metrics = PrometheusMetrics(app)

# Add custom metrics
metrics.info('app_info', 'Application info', version='1.0.0', model=MODEL_NAME)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive patient data and return heart disease prediction."""
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    logger.info(f"[{request_id}] Received prediction request")
    
    try:
        # Get JSON data from request body
        data = request.get_json(force=True)
        logger.info(f"[{request_id}] Input data received with {len(data)} features")
        
        # Convert data to DataFrame (required for model input)
        # Note: Input data structure must match training data
        input_df = pd.DataFrame([data])
        logger.debug(f"[{request_id}] Data converted to DataFrame: {input_df.shape}")
        
        # Make prediction using the model
        prediction = loaded_model.predict(input_df)
        prediction_proba = loaded_model.predict_proba(input_df)
        
        # Prediction output will be 0 or 1
        result = int(prediction[0])
        # Confidence score for the predicted class
        confidence_score = float(prediction_proba[0][result])
        
        logger.info(f"[{request_id}] ✓ Prediction successful: result={result}, confidence={confidence_score:.2f}")

        return jsonify({
            'prediction': result,
            'status': 'Prediction successful',
            'confidence_score': round(confidence_score, 2)
        })
    
    except KeyError as e:
        logger.error(f"[{request_id}] ✗ Missing required field: {str(e)}")
        return jsonify({
            'error': f'Missing required field: {str(e)}',
            'status': 'Prediction failed'
        }), 400
    
    except ValueError as e:
        logger.error(f"[{request_id}] ✗ Invalid data value: {str(e)}")
        return jsonify({
            'error': f'Invalid data value: {str(e)}',
            'status': 'Prediction failed'
        }), 400
    
    except Exception as e:
        logger.error(f"[{request_id}] ✗ Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'Prediction failed'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return jsonify({'status': 'ok', 'model': MODEL_NAME})


if __name__ == '__main__':
    # Set host to 0.0.0.0 for Docker access
    logger.info(f"Starting Flask application on 0.0.0.0:8080")
    logger.info(f"Model: {MODEL_NAME}")
    app.run(host='0.0.0.0', port=8080)
