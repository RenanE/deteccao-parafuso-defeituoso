"""One-Class SVM RBF com limiar externo calibrado em normais independentes."""

from numbers import Real
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_array, check_is_fitted

from .anomaly import CalibratedAnomalyMixin, _ids, _taxa


class CalibratedOneClassSVM(CalibratedAnomalyMixin, OutlierMixin, BaseEstimator):
    """gamma='scale' calculado apenas no fit; multiplicador 1 preserva o histórico.

    nu regula o ajuste do SVM, não a FPR operacional. O limiar nativo continua
    disponível em model_, mas predict usa exclusivamente a calibração externa.
    """

    def __init__(self, nu=0.05, gamma_multiplier=1.0, target_fpr=0.05):
        self.nu = nu
        self.gamma_multiplier = gamma_multiplier
        self.target_fpr = target_fpr

    def fit(self, X, y=None, *, sample_ids=None):
        self._reset_calibration()
        if isinstance(self.nu, bool) or not isinstance(self.nu, Real) or not 0 < self.nu <= 1:
            raise ValueError("nu deve estar em (0, 1].")
        if (isinstance(self.gamma_multiplier, bool) or not isinstance(self.gamma_multiplier, Real)
                or not np.isfinite(self.gamma_multiplier) or self.gamma_multiplier <= 0):
            raise ValueError("gamma_multiplier deve ser finito e positivo.")
        _taxa(self.target_fpr)
        X = check_array(X, dtype=np.float64, ensure_min_samples=2)
        if y is not None and (np.asarray(y).shape != (len(X),) or not np.all(np.asarray(y) == 1)):
            raise ValueError("Se fornecido, y deve conter apenas +1 (normal).")
        ids = _ids(sample_ids, len(X))
        variance = X.var()
        scale = 1 / (X.shape[1] * variance) if variance else 1.0
        gamma = "scale" if self.gamma_multiplier == 1 else float(scale * self.gamma_multiplier)
        model = OneClassSVM(kernel="rbf", gamma=gamma, nu=self.nu, cache_size=128).fit(X)
        if model.fit_status_ != 0:
            raise ValueError("O SVM não convergiu; configuração não pode ser utilizada.")
        self.model_ = model
        self.resolved_gamma_ = float(model._gamma)
        self.n_features_in_ = X.shape[1]
        self.fit_ids_ = ids
        return self

    def anomaly_score(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Número de características diferente do ajuste.")
        scores = -self.model_.score_samples(X)
        if not np.isfinite(scores).all():
            raise ValueError("Scores SVM não finitos.")
        return scores
