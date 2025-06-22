import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from sklearn.utils import Bunch
from sklearn.datasets import (
    load_iris, load_digits, load_wine, load_breast_cancer
)

def load_dataset(name: str) -> Bunch:
    """Load a dataset by name from sklearn."""
    datasets = {
        'iris': load_iris,
        'digits': load_digits,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not supported.")
    return datasets[name]()

def save_dataset_as_csv(dataset: Bunch, output_path: str) -> None:
    """Convert sklearn dataset to CSV and save."""
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ CSV saved to: {output_path}")

def train_and_save_model(
    data, target, model, output_path: str, test_size=0.2
) -> None:
    """Train and save a classification model."""
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=42
    )
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {output_path}")

if __name__ == "__main__":
    # Choose dataset name and paths
    dataset_name = 'breast_cancer'  # change to 'digits', 'wine', etc.
    dataset = load_dataset(dataset_name)

    csv_path = f'data/{dataset_name}/{dataset_name}_dataset.csv'
    model_path = f'data/{dataset_name}/{dataset_name}_model.pkl'

    # Save CSV
    save_dataset_as_csv(dataset, csv_path)

    # Train and save model
    train_and_save_model(dataset.data, dataset.target, LogisticRegression(max_iter=10000), model_path)
    print("✅ Model and dataset generation completed for :", dataset_name)