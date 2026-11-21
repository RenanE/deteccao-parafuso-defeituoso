"""Representações padronizadas, com PCA opcional completo ou só do HOG."""

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class InvalidPCADimension(ValueError):
    """Configuração impossível para o tamanho/posto efetivo do ajuste."""


class FeatureRepresentation(TransformerMixin, BaseEstimator):
    """fit recebe apenas o ajuste interno. Nunca faz seleção de configuração.

    Sem PCA: atributos padronizados na ordem original.
    PCA completo: componentes do vetor inteiro.
    PCA HOG: componentes do HOG seguidos pelos 33 outros atributos padronizados.
    Sem whitening e sem nova padronização após o PCA.
    """

    def __init__(self, hog_dimension, pca_scope="none", n_components=None, seed=42):
        self.hog_dimension = hog_dimension
        self.pca_scope = pca_scope
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y=None):
        self.__dict__.pop("fit_complete_", None)
        X = check_array(X, dtype=np.float64, ensure_min_samples=2)
        if isinstance(self.hog_dimension, bool) or not isinstance(self.hog_dimension, Integral) or self.hog_dimension < 1:
            raise ValueError("hog_dimension deve ser inteiro positivo.")
        if X.shape[1] != self.hog_dimension + 33:
            raise ValueError("Esperados atributos HOG seguidos de exatamente 33 atributos auxiliares.")
        if self.pca_scope not in ("none", "full", "hog"):
            raise ValueError("pca_scope deve ser none, full ou hog.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**32:
            raise ValueError("seed inválida.")
        if self.pca_scope == "none":
            if self.n_components is not None:
                raise ValueError("n_components deve ser None quando não há PCA.")
        elif isinstance(self.n_components, bool) or not isinstance(self.n_components, Integral) or self.n_components < 1:
            raise InvalidPCADimension("n_components deve ser inteiro positivo.")
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.pca_ = None
        self.effective_rank_ = None
        self.explained_variance_ = None
        self.n_output_features_ = X.shape[1]
        if self.pca_scope != "none":
            block = Z if self.pca_scope == "full" else Z[:, :self.hog_dimension]
            bound = min(len(block) - 1, block.shape[1])
            if self.n_components > bound:
                raise InvalidPCADimension(f"{self.n_components} componentes excedem o limite n_fit-1/dimensão ({bound}).")
            if not np.any(block):
                raise InvalidPCADimension("O ajuste tem posto zero; PCA não é aplicável.")
            # SVD completa calcula o posto numérico no ajuste, sem consultar
            # calibração/validação. As componentes não utilizadas são descartadas.
            pca = PCA(n_components=None, svd_solver="full", whiten=False, random_state=self.seed).fit(block)
            tol = np.finfo(block.dtype).eps * max(block.shape) * pca.singular_values_[0]
            rank = int(np.sum(pca.singular_values_ > tol))
            self.effective_rank_ = rank
            if self.n_components > rank:
                raise InvalidPCADimension(f"{self.n_components} componentes excedem o posto numérico do ajuste ({rank}).")
            n = self.n_components
            discarded = pca.explained_variance_[n:]
            pca.noise_variance_ = float(discarded.mean()) if len(discarded) else 0.0
            for attr in ("components_", "explained_variance_", "explained_variance_ratio_", "singular_values_"):
                setattr(pca, attr, getattr(pca, attr)[:n].copy())
            pca.n_components_ = n
            pca.n_components = n
            self.pca_ = pca
            self.explained_variance_ = float(pca.explained_variance_ratio_.sum())
            self.n_output_features_ = n + (33 if self.pca_scope == "hog" else 0)
        self.fit_complete_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "fit_complete_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Dimensão de entrada difere do ajuste.")
        Z = self.scaler_.transform(X)
        if self.pca_scope == "none":
            return Z
        if self.pca_scope == "full":
            return self.pca_.transform(Z)
        return np.hstack((self.pca_.transform(Z[:, :self.hog_dimension]), Z[:, self.hog_dimension:]))

    def state_nbytes(self):
        """Soma dos arrays retidos; não representa RSS/pico do processo."""
        check_is_fitted(self, "scaler_")
        return sum(value.nbytes for obj in (self.scaler_, self.pca_) if obj is not None
                   for value in vars(obj).values() if isinstance(value, np.ndarray))
