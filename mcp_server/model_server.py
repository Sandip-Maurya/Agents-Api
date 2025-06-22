from fastmcp import FastMCP
from typing import Any
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
import shap
import base64
import io
import matplotlib.pyplot as plt


mcp = FastMCP("ModelServer")

@mcp.tool

def mcp_f1_score_tool(data_dir: str) -> float:
    """
    Loads a trained classification model from a pickle file and returns its macro F1 score
    on the test split of the dataset (CSV with a 'target' column).

    Args:
        data_dir (str): Directory name for both model and dataset. Should match:
                        'iris', 'digits', 'wine', or 'breast_cancer'.

    Returns:
        float: Macro-averaged F1 score.
    """
    model_path = os.path.join('data', data_dir, f'{data_dir}_model.pkl')
    data_path = os.path.join('data', data_dir, f'{data_dir}_dataset.csv')

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load dataset
    df = pd.read_csv(data_path)
    target = df['target']
    features = df.drop(columns=['target'], errors='ignore')

    # Split and evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    y_pred = model.predict(X_test.values)
    score = f1_score(y_test, y_pred, average='macro')

    print(f"✅ F1 Score ({data_dir}): {score:.4f}")
    return score # type: ignore

@mcp.tool
def mcp_shap_summary_tool(data_dir: str) -> str:
    """
    Generates a SHAP summary plot for the model and dataset under `data_dir`,
    encodes it into a base64 string, and returns it.

    Args:
        data_dir (str): Name of the dataset directory (e.g., 'iris', 'digits').

    Returns:
        str: Base64-encoded string of the SHAP summary plot image.
    """
    model_path = os.path.join('data', data_dir, f'{data_dir}_model.pkl')
    data_path = os.path.join('data', data_dir, f'{data_dir}_dataset.csv')

    # Load model and data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(data_path)
    target = df['target']
    features = df.drop(columns=['target'], errors='ignore')

    # Split and use training data for SHAP (to simulate model's view)
    X_train, _, _, _ = train_test_split(features, target, test_size=0.2, random_state=42)

    # Choose appropriate SHAP explainer
    if hasattr(model, "predict_proba"):
        explainer = shap.Explainer(model, X_train)  # uses KernelExplainer or model-specific
    else:
        explainer = shap.Explainer(model.predict, X_train)

    shap_values = explainer(X_train)

    # Create SHAP summary plot and encode it
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    print("✅ SHAP summary plot generated and encoded.")
    return img_base64



if __name__ == "__main__":
    mcp.run(
        transport='streamable-http',
        port=8001,
    )