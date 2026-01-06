# MLOPS_Heart_Disease/src/train_model.py

from eda import run_eda
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from data_processor import load_data, split_data, create_preprocessor


def train_and_log_model(model, X_train, y_train, X_test, y_test,
                        preprocessor, model_name):
    """
    Trains the model with preprocessing pipeline and logs metrics to MLflow.
    """
    # Poora MLOps Pipeline banao (Preprocessor + Model)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               (model_name.lower(), model)])

    with mlflow.start_run(run_name=model_name):

        # 1. Model Train Karo
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # y_proba for AUC calculation (zaroori)
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        # 2. Metrics Calculate Karo
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        # 3. Logging (Parameters aur Metrics)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)

        # 4. Model Log Karo (Task 4: Packaging complete)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            registered_model_name=f"{model_name}_Heart_Classifier"
        )

        print(f"--- {model_name} Logged to MLflow. ROC AUC: {auc:.4f} ---")
        return auc


def run_training():
    """Main function to setup MLflow and run model training."""

    # 1. MLflow Setup
    mlflow.set_tracking_uri("./mlruns")
    experiment_name = "Heart_Disease_Prediction_Assignment"

    # Experiment check aur set
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # 2. Data Preparation
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    preprocessor = create_preprocessor(data)

    # 3.  EDA 
    run_eda(data)

    # 4. Model Definitions
    lr = LogisticRegression(solver='liblinear', random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 5. Training and Logging
    lr_auc = train_and_log_model(lr, X_train, y_train, X_test, y_test,
                                 preprocessor, "LogisticRegression")
    rf_auc = train_and_log_model(rf, X_train, y_train, X_test, y_test,
                                 preprocessor, "RandomForest")

    # 6. Final Decision
    if lr_auc > rf_auc:
        print(f"\nFinal Decision: Logistic Regression is the best model "
              f"(ROC AUC: {lr_auc:.4f}).")
    else:
        print(f"\nFinal Decision: Random Forest is the best model "
              f"(ROC AUC: {rf_auc:.4f}).")


if __name__ == '__main__':
    run_training()
# End of src/train_model.py
