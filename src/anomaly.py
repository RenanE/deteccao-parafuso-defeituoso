"""Calibração independente compartilhada pelos detectores One-Class.

Scores maiores em anomaly_score indicam anomalia. score_samples tem sinal
oposto; decision_function >= 0 e predict=+1 indicam normalidade.
"""
from numbers import Real
import warnings
import numpy as np
from sklearn.utils.validation import check_is_fitted

def _ids(sample_ids, tamanho):
    if sample_ids is None:
        return None
    valores = tuple(sample_ids)
    if len(valores) != tamanho or any(not isinstance(x, str) or not x.strip() for x in valores):
        raise ValueError("sample_ids deve conter um identificador textual por amostra.")
    if len(set(valores)) != tamanho:
        raise ValueError("sample_ids contém identificadores duplicados.")
    return valores


def _taxa(valor):
    if isinstance(valor, bool) or not isinstance(valor, Real) or not 0 < valor < 1:
        raise ValueError("target_fpr deve estar estritamente entre 0 e 1.")
    return float(valor)


class CalibratedAnomalyMixin:
    def _reset_calibration(self):
        for name in ("threshold_", "calibration_scores_", "calibration_fpr_",
                     "target_fpr_", "n_calibration_", "calibration_ids_"):
            self.__dict__.pop(name, None)

    def calibrate(self, X, y=None, *, sample_ids=None, target_fpr=None):
        """Quantil empírico superior (numpy method='higher') de boas reservadas.

        Empates no limiar são normais. A FPR empírica é conservadora; não é uma
        garantia de FPR futura/populacional nem uma estimativa de recall.
        """
        scores = self.anomaly_score(X)
        if len(scores) < 2:
            raise ValueError("A calibração exige pelo menos duas amostras normais independentes.")
        if y is not None and (np.asarray(y).shape != (len(scores),) or not np.all(np.asarray(y) == 1)):
            raise ValueError("Calibração aceita apenas y=+1 (normal).")
        ids = _ids(sample_ids, len(scores))
        if self.fit_ids_ is not None:
            if ids is None:
                raise ValueError("Forneça sample_ids na calibração para verificar independência do ajuste.")
            if set(ids) & set(self.fit_ids_):
                raise ValueError("Vazamento: IDs de calibração presentes no ajuste.")
        alpha = _taxa(self.target_fpr if target_fpr is None else target_fpr)
        if len(scores) * alpha < 1:
            warnings.warn("Calibração pequena: 1/n excede a FPR-alvo; o quantil pode selecionar o máximo.", UserWarning, stacklevel=2)
        self.threshold_ = float(np.quantile(scores, 1 - alpha, method="higher"))
        self.calibration_scores_ = scores.copy()
        self.calibration_fpr_ = float(np.mean(scores > self.threshold_))
        self.target_fpr_ = alpha
        self.n_calibration_ = len(scores)
        self.calibration_ids_ = ids
        return self

    def score_samples(self, X):
        return -self.anomaly_score(X)

    def decision_function(self, X):
        check_is_fitted(self, "threshold_")
        return self.threshold_ + self.score_samples(X)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
