"""Etapa 9: descritores LOCAIS clássicos sobre a ROI alinhada.

Diagnóstico da etapa 8: o vetor global é 99,9% HOG e o HOG global não separa
defeitos (AUC ~0,5). Defeitos de parafuso são pequenos e locais; resumir a peça
inteira num vetor dilui o sinal. Como a ROI já sai alinhada (cabeça à esquerda,
eixo na horizontal), cada posição da grade corresponde aproximadamente à mesma
parte do parafuso em todas as imagens. Assim:

1. A ROI 384x128 é dividida numa grade de patches sobrepostos.
2. Cada patch recebe um descritor clássico curto (HOG local, LBP, energia de
   Gabor, densidade de bordas, fração de máscara).
3. Para cada posição, um detector One-Class (SVM, LOF ou Isolation Forest) é
   ajustado com os patches das boas naquela posição e nas vizinhas (tolerância a
   pequenos desalinhamentos e à fase da rosca).
4. O score do patch é padronizado pela distribuição de scores de boas fora da
   amostra (ajuste cruzado em 2 dobras) e o score da imagem é o maior score
   entre seus patches (ou a média dos k maiores).

Tudo é visão clássica: nenhum aprendizado profundo.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .features import criar_banco_gabor
from .preprocessing import aplicar_clahe

DESCRIPTORS = ("hog", "lbp", "gabor", "edges", "shape")


@dataclass(frozen=True)
class LocalConfig:
    patch: int = 32                     # lado do patch (pixels na ROI 384x128)
    descriptors: tuple = DESCRIPTORS
    znorm: bool = True                  # normaliza brilho/contraste por imagem
    neighbors: int = 1                  # posições vizinhas (cada lado) no ajuste
    aggregation: str = "max"            # "max" ou "top3"

    def key(self):
        return f"P{self.patch}-{'+'.join(self.descriptors)}-z{int(self.znorm)}-n{self.neighbors}-{self.aggregation}"


def preparar(roi, mask, znorm):
    img = aplicar_clahe(roi, clip_limit=2.0, tile_grid_size=(8, 8)).astype(np.float32)
    if znorm:
        inside = img[mask > 0] if mask.any() else img.ravel()
        img = (img - inside.mean()) / (inside.std() + 1e-6) * 40 + 128
        img = np.clip(img, 0, 255)
    return img


def descritores_imagem(roi, mask, config, bank):
    """Retorna array (linhas, colunas, d) com um descritor por patch."""
    P, S = config.patch, config.patch // 2
    img = preparar(roi, mask, config.znorm)
    H, W = img.shape
    rows, cols = (H - P) // S + 1, (W - P) // S + 1
    maps = {}
    if "hog" in config.descriptors:
        # Blocos 2x2 de células P/2: cada bloco coincide com um patch da grade.
        h = hog(img, orientations=9, pixels_per_cell=(S, S), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=False)
        maps["hog"] = h.reshape(h.shape[0], h.shape[1], -1)[:rows, :cols]
    lbp = local_binary_pattern(img.astype(np.uint8), 8, 1, "uniform") if "lbp" in config.descriptors else None
    gab = ([np.abs(cv2.filter2D(img, cv2.CV_32F, k)) for k in bank]
           if "gabor" in config.descriptors else None)
    edges = cv2.Canny(img.astype(np.uint8), 50, 150) > 0 if "edges" in config.descriptors else None
    out = np.zeros((rows, cols, 0), np.float32)
    feats = []
    for r in range(rows):
        row = []
        for c in range(cols):
            y, x = r * S, c * S
            parts = []
            if "hog" in config.descriptors:
                parts.append(maps["hog"][r, c])
            if lbp is not None:
                hist = np.bincount(lbp[y:y + P, x:x + P].astype(int).ravel(), minlength=10)[:10]
                parts.append(hist / hist.sum())
            if gab is not None:
                parts.append([g[y:y + P, x:x + P].mean() for g in gab])
            if edges is not None:
                parts.append([edges[y:y + P, x:x + P].mean()])
            if "shape" in config.descriptors:
                m = mask[y:y + P, x:x + P]
                ys = np.where(m.any(1))[0]
                parts.append([m.mean(), (ys.min() / P) if len(ys) else 1.0, (ys.max() / P) if len(ys) else 0.0])
            row.append(np.concatenate([np.asarray(p, np.float32).ravel() for p in parts]))
        feats.append(row)
    out = np.asarray(feats, np.float32)
    return out


def extrair_lote(rois, masks, config, n_jobs=-1):
    bank = criar_banco_gabor(orientations=4, lambdas=[8.0, 12.0, 16.0])
    arrays = Parallel(n_jobs=n_jobs)(delayed(descritores_imagem)(r, m, config, bank)
                                     for r, m in zip(rois, masks))
    return np.stack(arrays)  # (n, linhas, colunas, d)


def make_detector(model, params, seed=42):
    if model == "svm":
        params = {"gamma": "scale", **params}
        return OneClassSVM(kernel="rbf", **params)
    if model == "lof":
        return LocalOutlierFactor(novelty=True, **params)
    if model == "isolation_forest":
        return IsolationForest(random_state=seed, n_jobs=1, **params)
    raise ValueError(model)


def _score(det, X):
    # Convenção do projeto: maior = mais anômalo.
    return -det.score_samples(X) if not isinstance(det, OneClassSVM) else -det.decision_function(X)


def _pool(F, r, c, k):
    rows, cols = F.shape[1], F.shape[2]
    blocks = [F[:, rr, cc] for rr in range(max(0, r - k), min(rows, r + k + 1))
              for cc in range(max(0, c - k), min(cols, c + k + 1))]
    return np.concatenate(blocks)


def _fit_position(Xtr, Xref, model, params, seed, targets=()):
    """Ajusta uma posição; devolve média/desvio de referência e os scores brutos dos alvos.

    O modelo não é devolvido ao processo principal (economia de memória: centenas de
    florestas por configuração não cabem na RAM).
    """
    scaler = StandardScaler().fit(Xtr)
    det = make_detector(model, params, seed).fit(scaler.transform(Xtr))
    ref = _score(det, scaler.transform(Xref))
    return float(ref.mean()), float(ref.std()), [_score(det, scaler.transform(X)) for X in targets]


class PatchAnomalyDetector:
    """Detector local: um modelo clássico por posição da grade; score = max dos patches."""

    def __init__(self, model, params, config: LocalConfig, seed=42, n_jobs=-1):
        self.model, self.params, self.config, self.seed, self.n_jobs = model, params, config, seed, n_jobs

    def fit_score(self, F, targets):
        """Ajusta nas boas F e devolve mapas de score (n, linhas, colunas) para cada alvo.

        Ajuste cruzado em 2 dobras: cada metade das boas ajusta um modelo por posição
        (patches da posição e das vizinhas), a outra metade, fora da amostra, dá a
        média/desvio que padronizam o score daquela posição. O score de um patch é a
        média dos dois scores padronizados. Os modelos não são guardados.
        """
        n, rows, cols, _ = F.shape
        self.shape_ = (rows, cols)
        order = np.random.default_rng(self.seed).permutation(n)
        halves = (order[: n // 2], order[n // 2:])
        positions = [(r, c) for r in range(rows) for c in range(cols)]
        k = self.config.neighbors
        maps = [np.zeros((len(T), rows, cols)) for T in targets]
        for a, b in (halves, halves[::-1]):
            Fa, Fb = F[a], F[b]
            res = Parallel(n_jobs=self.n_jobs, batch_size=8)(
                delayed(_fit_position)(_pool(Fa, r, c, k), Fb[:, r, c], self.model, self.params, self.seed,
                                       [T[:, r, c] for T in targets])
                for r, c in positions)
            # Piso do desvio: posições quase constantes (fundo liso) não podem ter z-scores explosivos.
            floor = max(0.5 * float(np.median([sd for _, sd, _ in res])), 1e-9)
            for (r, c), (mu, sd, scores) in zip(positions, res):
                for m, s_ in zip(maps, scores):
                    m[:, r, c] += (s_ - mu) / max(sd, floor)
        return [m / 2 for m in maps]

    def aggregate(self, maps):
        flat = maps.reshape(len(maps), -1)
        if self.config.aggregation == "max":
            return flat.max(1)
        return np.sort(flat, 1)[:, -3:].mean(1)

    def describe(self):
        return {"model": self.model, "params": self.params, "config": asdict(self.config), "seed": self.seed}
