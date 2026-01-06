import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Headers for the Cleveland dataset (14 columns)
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]


def load_data(path: str = "data/processed.cleveland.data") -> pd.DataFrame:
    """Loads and returns the cleaned heart disease dataset."""
    try:
        # Load CSV with custom column names and mark '?' as missing values
        df = pd.read_csv(path, header=None, names=COLUMNS, na_values=['?'])
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}. Please ensure data is "
              f"present.")
        raise

    # Remove rows with missing values
    df = df.dropna()
    
    # Convert target to binary: 0 = no disease, 1 = disease (any stage)
    df['target'] = df['target'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42):
    """Splits data into training and testing sets with stratification."""
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def create_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Creates the preprocessing ColumnTransformer (scaling and encoding)."""

    # Continuous numerical features
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Categorical features to be one-hot encoded
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang',
                            'slope', 'ca', 'thal']

    # Standardize numerical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # One-hot encode categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


if __name__ == '__main__':
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    preprocessor = create_preprocessor(data)
    print(f"data_processor.py: Data loaded ({data.shape[0]} rows) and split "
          f"successfully.")