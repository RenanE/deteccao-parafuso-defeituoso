"""Isolation Forest com random_state fixo e calibração externa independente."""

from numbers import Integral, Real
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_array, check_is_fitted

from .anomaly import CalibratedAnomalyMixin, _ids, _taxa


class IsolationForestAnomalyDetector(CalibratedAnomalyMixin, OutlierMixin, BaseEstimator):
    def __init__(self, n_estimators=100, max_samples=128, max_features=1.0,
                 random_state=42, target_fpr=0.05):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.target_fpr = target_fpr

    def fit(self, X, y=None, *, sample_ids=None):
        self._reset_calibration()
        self.__dict__.pop("model_", None)
        X = check_array(X, dtype=np.float64, ensure_min_samples=3)
        n, p = X.shape
        for name in ("n_estimators", "random_state"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (1 if name == "n_estimators" else 0):
                raise ValueError(f"{name} deve ser inteiro válido.")
        if self.random_state >= 2**32:
            raise ValueError("random_state fora do intervalo permitido.")
        if isinstance(self.max_samples, bool):
            raise ValueError("max_samples inválido.")
        if isinstance(self.max_samples, Integral):
            valid = 2 <= self.max_samples <= n
        elif isinstance(self.max_samples, Real):
            valid = 0 < self.max_samples <= 1 and int(self.max_samples * n) >= 2
        else:
            valid = self.max_samples == "auto"
        if not valid:
            raise ValueError("max_samples deve selecionar entre 2 e n amostras; sem clipping silencioso.")
        if isinstance(self.max_features, bool) or not isinstance(self.max_features, Real):
            raise ValueError("max_features inválido.")
        valid = (1 <= self.max_features <= p if isinstance(self.max_features, Integral)
                 else 0 < self.max_features <= 1)
        if not valid:
            raise ValueError("max_features fora do intervalo permitido.")
        _taxa(self.target_fpr)
        if y is not None and (np.asarray(y).shape != (n,) or not np.all(np.asarray(y) == 1)):
            raise ValueError("Ajuste aceita apenas y=+1 (normal).")
        if np.all(X == X[0]):
            raise ValueError("Ajuste degenerado: todas as amostras são idênticas.")
        ids = _ids(sample_ids, n)
        model = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                max_features=self.max_features, random_state=self.random_state,
                                contamination="auto", n_jobs=1).fit(X)
        self.model_ = model
        self.fit_ids_ = ids
        self.n_features_in_ = p
        self.diagnostics_ = {"effective_max_samples": int(model.max_samples_),
                             "features_per_tree": len(model.estimators_features_[0]),
                             "trees": len(model.estimators_), "random_state": self.random_state,
                             "native_offset_ignored": float(model.offset_)}
        return self

    def anomaly_score(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Número de características diferente do ajuste.")
        scores = -self.model_.score_samples(X)
        if not np.isfinite(scores).all():
            raise ValueError("Scores Isolation Forest não finitos.")
        return scores
