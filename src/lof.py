"""Local Outlier Factor (modo novelty) com calibração externa independente.

Mesma convenção dos demais detectores: `anomaly_score` cresce com a anomalia,
`score_samples` tem sinal oposto e `predict` devolve +1 (normal) / -1 (defeito).
O limiar nativo do LOF (`offset_`) é ignorado; o limiar operacional vem de
`calibrate`, usando somente boas reservadas.
"""

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_array, check_is_fitted

from .anomaly import CalibratedAnomalyMixin, _ids, _taxa


class LOFAnomalyDetector(CalibratedAnomalyMixin, OutlierMixin, BaseEstimator):
    """LOF ajustado apenas em boas; score = -score_samples do LOF (fator local)."""

    def __init__(self, n_neighbors=20, metric="euclidean", target_fpr=0.05):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.target_fpr = target_fpr

    def fit(self, X, y=None, *, sample_ids=None):
        self._reset_calibration()
        self.__dict__.pop("model_", None)
        X = check_array(X, dtype=np.float64, ensure_min_samples=3)
        n, p = X.shape
        k = self.n_neighbors
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= n - 1:
            raise ValueError("n_neighbors deve ser inteiro entre 1 e n_ajuste - 1; sem redução silenciosa.")
        _taxa(self.target_fpr)
        if y is not None and (np.asarray(y).shape != (n,) or not np.all(np.asarray(y) == 1)):
            raise ValueError("Ajuste aceita apenas y=+1 (normal).")
        if np.all(X == X[0]):
            raise ValueError("Ajuste degenerado: todas as amostras são idênticas.")
        ids = _ids(sample_ids, n)
        self.model_ = LocalOutlierFactor(n_neighbors=k, metric=self.metric, novelty=True,
                                         contamination="auto", n_jobs=1).fit(X)
        self.fit_ids_ = ids
        self.n_features_in_ = p
        # LOF de treino (leave-self-out pelo próprio algoritmo), apenas diagnóstico.
        self.training_scores_ = -self.model_.negative_outlier_factor_.copy()
        self.diagnostics_ = {"n_neighbors": k, "metric": self.metric,
                             "native_offset_ignored": float(self.model_.offset_)}
        return self

    def anomaly_score(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Número de características diferente do ajuste.")
        scores = -self.model_.score_samples(X)
        if not np.isfinite(scores).all():
            raise ValueError("Scores LOF não finitos.")
        return scores
