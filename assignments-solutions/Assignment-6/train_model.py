# train_model.py - Train and save Iris Random Forest model 🌸

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODEL_PATH = "iris_model.joblib"

def train_and_save_model():
    # Load Iris dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names = iris.target_names

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained successfully with accuracy: {acc:.2f}")

    # Save model with metadata
    artifacts = {
        "model": model,
        "feature_names": feature_names,
        "target_names": target_names,
    }
    joblib.dump(artifacts, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
