"""k-NN One-Class com ajuste e calibração independentes."""

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

from .anomaly import CalibratedAnomalyMixin, _ids, _taxa


class KNNAnomalyDetector(CalibratedAnomalyMixin, OutlierMixin, BaseEstimator):
    """Distância média ou ao k-ésimo vizinho; scores maiores são mais normais.

    fit não define limiar. calibrate exige normais independentes do ajuste.
    Use sample_ids em ambos os métodos para verificar a independência por ID;
    o protocolo em validation.py verifica também grupos e hashes disponíveis.
    Sem IDs, esta classe de baixo nível não pode identificar a origem dos vetores.
    """

    def __init__(self, n_neighbors=5, metric="euclidean", target_fpr=0.05, aggregation="mean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.target_fpr = target_fpr
        self.aggregation = aggregation

    def fit(self, X, y=None, *, sample_ids=None):
        if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, Integral) or self.n_neighbors < 1:
            raise ValueError("n_neighbors deve ser um inteiro positivo.")
        self._reset_calibration()
        if self.aggregation not in ("mean", "kth"):
            raise ValueError("aggregation deve ser mean ou kth.")
        _taxa(self.target_fpr)
        if self.metric == "precomputed":
            raise ValueError("Use vetores de características, não distâncias precomputed.")
        X = check_array(X, dtype=np.float64, copy=True)
        if len(X) <= self.n_neighbors:
            raise ValueError("O ajuste precisa de pelo menos n_neighbors + 1 amostras para excluir a autovizinhança.")
        if y is not None and (np.asarray(y).shape != (len(X),) or not np.all(np.asarray(y) == 1)):
            raise ValueError("Se fornecido, y deve conter apenas +1 (normal).")
        ids = _ids(sample_ids, len(X))
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(X)
        # X=None exclui a própria amostra por índice, mantendo zeros legítimos
        # entre amostras distintas com vetores iguais.
        distances, indices = nn.kneighbors()
        if not np.isfinite(distances).all():
            raise ValueError("Distâncias de ajuste não finitas.")
        self.nn_ = nn
        self.n_features_in_ = X.shape[1]
        self.fit_ids_ = ids
        self.training_neighbor_indices_ = indices
        self.training_scores_ = self._aggregate(distances)
        return self

    def _aggregate(self, distances):
        return distances.mean(axis=1) if self.aggregation == "mean" else distances[:, -1]

    def anomaly_score(self, X):
        """Distância agregada: maior significa mais anômalo; não requer limiar."""
        check_is_fitted(self, "nn_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Número de características diferente do ajuste.")
        distances, _ = self.nn_.kneighbors(X)
        if not np.isfinite(distances).all():
            raise ValueError("Distâncias não finitas.")
        return self._aggregate(distances)

