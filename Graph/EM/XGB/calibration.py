"""Calibration utilities for Co-EM expert probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression


CalibrationMethod = Literal["platt", "temperature", "none"]


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: CalibrationMethod
    coef: float = 1.0
    intercept: float = 0.0
    temperature: float = 1.0

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        values = _as_finite_1d(logits, "logits")
        if self.method == "platt":
            return values * self.coef + self.intercept
        if self.method == "temperature":
            return values / self.temperature
        if self.method == "none":
            return values
        raise ValueError(f"unknown calibration method: {self.method}")

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return expit(self.transform_logits(logits)).astype(np.float64, copy=False)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "method": self.method,
            "coef": float(self.coef),
            "intercept": float(self.intercept),
            "temperature": float(self.temperature),
        }


def fit_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    method: CalibrationMethod = "platt",
) -> ProbabilityCalibrator:
    values = _as_finite_1d(logits, "logits")
    y = np.asarray(labels, dtype=np.int8)
    if values.shape[0] != y.shape[0]:
        raise ValueError("logits and labels must have the same length")
    if len(np.unique(y)) < 2:
        raise ValueError("calibration requires validation labels from both classes")
    if method == "none":
        return ProbabilityCalibrator(method="none")
    if method == "platt":
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(values.reshape(-1, 1), y)
        return ProbabilityCalibrator(
            method="platt",
            coef=float(model.coef_[0, 0]),
            intercept=float(model.intercept_[0]),
        )
    if method == "temperature":
        result = minimize_scalar(
            lambda log_temp: _binary_cross_entropy(values / np.exp(log_temp), y),
            bounds=(-5.0, 5.0),
            method="bounded",
        )
        if not result.success:
            raise RuntimeError(f"temperature calibration failed: {result.message}")
        return ProbabilityCalibrator(
            method="temperature",
            temperature=float(np.exp(result.x)),
        )
    raise ValueError(f"unknown calibration method: {method}")


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    probs = _as_finite_1d(probabilities, "probabilities")
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return logit(clipped).astype(np.float64, copy=False)


def _as_finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _binary_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(expit(logits), 1e-7, 1.0 - 1e-7)
    y = labels.astype(np.float64, copy=False)
    return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log1p(-probs)))
