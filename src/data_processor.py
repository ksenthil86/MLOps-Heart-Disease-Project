# MLOPS_Heart_Disease/src/data_processor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from eda import run_eda

# Headers for the Cleveland dataset (14 columns)
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]


def load_data(path: str = "data/processed.cleveland.data") -> pd.DataFrame:
    """Loads and returns the cleaned heart disease dataset."""
    try:
        # header=None: Kyunki raw data mein header nahi hai.
        # names=COLUMNS: Sahi column names assign karne ke liye.
        # na_values=['?']: Missing values ko NaN mein badalne ke liye.
        df = pd.read_csv(path, header=None, names=COLUMNS, na_values=['?'])
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}. Please ensure data is "
              f"present.")
        raise

    # Missing values (jo '?' the) ko drop karo
    df = df.dropna()

    # Target column ko 0s aur 1s mein clean karo (4 values of disease mapping
    # to 1)  <-- E501 FIX
    df['target'] = df['target'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42):  # <-- E501 FIX
    """Splits data into training and testing sets with stratification."""
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def create_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Creates the preprocessing ColumnTransformer (scaling and encoding)."""

    # Numeric features
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    # E501 Fix: Line ko chota kiya
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang',
                            'slope', 'ca', 'thal']

    # Transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


if __name__ == '__main__':
    # Test run
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    preprocessor = create_preprocessor(data)
    print(f"data_processor.py: Data loaded ({data.shape[0]} rows) and split "
          f"successfully.")
    print(f"run EDA on data")
    run_eda(data)
    

