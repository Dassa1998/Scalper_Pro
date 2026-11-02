from __future__ import annotations

import logging
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

log = logging.getLogger(__name__)


class AIModel:
    """Lightweight ML model predicting P(TP before SL). Filters trades."""

    def __init__(self, model_path: str = "ai_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.metrics = {"accuracy": 0.0, "trained_at": 0}

    def load(self) -> bool:
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data["model"]
                self.metrics = data.get("metrics", self.metrics)
                self.is_loaded = True
                log.info("AI model loaded successfully.")
                return True
            except Exception as e:
                log.error(f"Failed to load AI model: {e}")
        self.is_loaded = False
        return False

    def save(self) -> None:
        if self.model:
            joblib.dump({"model": self.model, "metrics": self.metrics}, self.model_path)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a simple logistic regression. Called in background thread."""
        if len(X) < 100 or len(np.unique(y)) < 2:
            log.warning("Not enough data to train AI model.")
            return
        try:
            clf = LogisticRegression(max_iter=1000)
            # Simple train-test split for metrics
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            
            self.model = clf
            self.metrics = {"accuracy": float(acc), "trained_at": int(os.times()[4])}
            self.is_loaded = True
            self.save()
            log.info(f"AI model trained. Accuracy: {acc:.2f}")
        except Exception as e:
            log.error(f"Training failed: {e}")

    def predict_proba(self, features: dict) -> float:
        """Returns probability of success (0.0 to 1.0)."""
        if not self.is_loaded or self.model is None:
            return 1.0  # Fallback: don't block trades if model is missing

        # Extract features in exact same order as training
        # In a real setup, define FEATURE_NAMES and map them
        try:
            # Example placeholder for feature vector construction
            # Must match length expected by model (e.g., 12 features)
            X = np.array([list(features.values())])
            prob = self.model.predict_proba(X)[0][1]  # prob of class 1
            return float(prob)
        except Exception as e:
            log.warning(f"AI predict error: {e}")
            return 1.0
