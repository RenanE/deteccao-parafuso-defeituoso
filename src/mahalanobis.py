"""Mahalanobis quadrática: covariância Ledoit-Wolf ou EllipticEnvelope viável."""

from numbers import Integral, Real
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.covariance import EllipticEnvelope, LedoitWolf
from sklearn.utils.validation import check_array, check_is_fitted

from .anomaly import CalibratedAnomalyMixin, _ids, _taxa


class NumericalFitError(ValueError):
    """Configuração inviável numericamente; registrar sem fallback silencioso."""


class MahalanobisAnomalyDetector(CalibratedAnomalyMixin, OutlierMixin, BaseEstimator):
    """anomaly_score=d²; score_samples=-d². Sem inversão/pseudoinversa silenciosa.

    Covariâncias limitadas a 256 atributos. Elliptic exige n>=2p+1, posto
    completo e condição <=1e12; isso não garante robustez estatística em p alto.
    O limiar nativo do EllipticEnvelope é descartado.
    """

    def __init__(self, covariance="ledoit_wolf", support_fraction=None,
                 random_state=42, target_fpr=0.05):
        self.covariance = covariance
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.target_fpr = target_fpr

    def fit(self, X, y=None, *, sample_ids=None):
        self._reset_calibration()
        self.__dict__.pop("fit_complete_", None)
        X = check_array(X, dtype=np.float64, ensure_min_samples=3)
        n, p = X.shape
        if p > 256:
            raise ValueError("Mahalanobis limitado a 256 atributos; use R2/R3, nunca R1.")
        if self.covariance not in ("ledoit_wolf", "elliptic"):
            raise ValueError("covariance deve ser ledoit_wolf ou elliptic.")
        if (isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state < 2**32):
            raise ValueError("random_state deve ser uma semente inteira válida.")
        if self.support_fraction is not None and (
                isinstance(self.support_fraction, bool) or not isinstance(self.support_fraction, Real)
                or not 0 < self.support_fraction <= 1):
            raise ValueError("support_fraction deve estar em (0,1] ou ser None.")
        if self.covariance == "ledoit_wolf" and self.support_fraction is not None:
            raise ValueError("support_fraction só se aplica ao EllipticEnvelope.")
        _taxa(self.target_fpr)
        if y is not None and (np.asarray(y).shape != (n,) or not np.all(np.asarray(y) == 1)):
            raise ValueError("Ajuste aceita apenas y=+1 (normal).")
        ids = _ids(sample_ids, n)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                centered = X - X.mean(axis=0)
                rank = int(np.linalg.matrix_rank(centered))
                if rank == 0:
                    raise NumericalFitError("Ajuste sem variabilidade: covariância de posto zero.")
                if self.covariance == "elliptic" and (n < 2 * p + 1 or rank < p):
                    raise NumericalFitError(f"Elliptic inviável: n={n}, p={p}, posto={rank}; exige n>=2p+1 e posto completo.")
                estimator = (LedoitWolf(store_precision=False) if self.covariance == "ledoit_wolf"
                             else EllipticEnvelope(contamination=.1, support_fraction=self.support_fraction,
                                                   random_state=self.random_state))
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter("always")
                    estimator.fit(X)
                if notices:
                    raise NumericalFitError("Avisos no ajuste de covariância: " + "; ".join(str(w.message) for w in notices))
                covariance = estimator.covariance_
                eigenvalues = np.linalg.eigvalsh(covariance)
                if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
                    raise NumericalFitError("Covariância não é positiva definida e finita.")
                condition = float(eigenvalues[-1] / eigenvalues[0])
                if condition > 1e12:
                    raise NumericalFitError(f"Covariância mal condicionada: {condition:.6g} > 1e12.")
                cholesky = np.linalg.cholesky(covariance)
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            raise NumericalFitError(f"Falha numérica na covariância: {exc}") from exc
        self.location_ = estimator.location_.copy()
        self.covariance_ = covariance.copy()
        self.cholesky_ = cholesky
        self.fit_ids_ = ids
        self.n_features_in_ = p
        self.diagnostics_ = {"input_rank": rank, "condition_number": condition,
                             "min_eigenvalue": float(eigenvalues[0]),
                             "max_eigenvalue": float(eigenvalues[-1]),
                             "shrinkage": float(estimator.shrinkage_) if self.covariance == "ledoit_wolf" else None,
                             "robust_support_size": int(estimator.support_.sum()) if self.covariance == "elliptic" else None}
        self.fit_complete_ = True
        return self

    def anomaly_score(self, X):
        check_is_fitted(self, "fit_complete_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Número de características diferente do ajuste.")
        try:
            with np.errstate(over="raise", invalid="raise"):
                whitened = solve_triangular(self.cholesky_, (X - self.location_).T, lower=True)
                distances = np.sum(whitened ** 2, axis=0)
        except (FloatingPointError, ValueError) as exc:
            raise ValueError(f"Falha numérica nos scores Mahalanobis: {exc}") from exc
        if not np.isfinite(distances).all():
            raise ValueError("Distâncias Mahalanobis não finitas.")
        return distances
