"""Divisões persistentes One-Class e transformações ajustadas só no ajuste.

Recebe metadados já validados pelo carregador; não presume rótulos de imagens.
Nenhuma função treina modelos ou extrai descritores de imagens.
"""

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array

from .dataset import CATEGORIAS_TESTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUP_COLUMNS = ("grupo", "peca_id", "aquisicao_id", "group_id", "piece_id", "acquisition_id")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _positivo(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} deve ser inteiro positivo.")


def normalizar_caminhos(df, *, base_dir=PROJECT_ROOT):
    """IDs canônicos; CSVs históricos usam a raiz do projeto por padrão."""
    result = df.copy()
    if "caminho" not in result:
        raise ValueError("Coluna caminho obrigatória.")
    ids = []
    for value in result["caminho"]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Cada caminho deve ser texto não vazio.")
        path = Path(value).expanduser()
        ids.append(str((path if path.is_absolute() else Path(base_dir) / path).resolve()))
    result["caminho"] = ids
    if result["caminho"].duplicated().any():
        raise ValueError("Caminhos duplicados no inventário, inclusive entre splits.")
    return result


def adicionar_hashes(df):
    """Lê bytes sem modificar imagens; use após confirmar os rótulos no carregador."""
    # O carregador resolve argumentos relativos a partir do diretório atual.
    result = normalizar_caminhos(df, base_dir=Path.cwd())
    hashes = []
    for path in result["caminho"]:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes.append(digest.hexdigest())
    result["sha256"] = hashes
    return result


def _catalogo(df, group_columns=None):
    if not isinstance(df, pd.DataFrame) or not {"caminho", "classe", "split"}.issubset(df.columns):
        raise ValueError("Informe DataFrame com caminho, classe e split.")
    if df.empty or not df.columns.is_unique:
        raise ValueError("Inventário vazio ou com colunas duplicadas.")
    frame = normalizar_caminhos(df)
    if frame[["classe", "split"]].isna().any().any():
        raise ValueError("Rótulos/splits ausentes.")
    if not frame["split"].isin(["train", "test"]).all():
        raise ValueError("Split inesperado; permitidos train e test.")
    if not frame["classe"].isin(CATEGORIAS_TESTE).all():
        raise ValueError("Classe desconhecida no inventário.")
    treino = frame["split"] == "train"
    if not treino.any() or not frame.loc[treino, "classe"].eq("good").all():
        raise ValueError("A divisão exige treino exclusivamente good, previamente confirmado.")
    if isinstance(group_columns, str):
        raise ValueError("group_columns deve ser uma lista de nomes de colunas.")
    grupos = sorted(set(group_columns or []) | {c for c in GROUP_COLUMNS if c in frame})
    if any(c not in frame or c in ("caminho", "classe", "split", "sha256") for c in grupos):
        raise ValueError("Coluna de grupo ausente ou reservada.")
    for col in grupos:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Identificador de grupo ausente: {col}.")
        frame[col] = frame[col].astype(str)
    if "sha256" in frame:
        if not frame["sha256"].map(lambda v: isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None).all():
            raise ValueError("sha256 deve conter hashes hexadecimais completos.")
    cols = ["caminho", "classe", "split"] + grupos + (["sha256"] if "sha256" in frame else [])
    frame = frame[cols].sort_values("caminho").reset_index(drop=True)
    return frame, grupos


def _componentes(frame, grupos):
    """Agrupa por peça/aquisição E duplicatas; relações transitivas são mantidas."""
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for col in grupos + (["sha256"] if "sha256" in frame else []):
        first = {}
        for i, value in enumerate(frame[col]):
            if value in first:
                a, b = find(i), find(first[value])
                parent[max(a, b)] = min(a, b)
            else:
                first[value] = i
    components = {}
    for i in range(len(frame)):
        components.setdefault(find(i), []).append(i)
    for members in components.values():
        if frame.iloc[members]["split"].nunique() > 1:
            raise ValueError("Vazamento: grupo ou hash compartilhado entre train e test.")
    return list(components.values())


def gerar_divisao(df, *, calibration_fraction=0.2, seed=42, group_columns=None,
                  min_fit=6, min_calibration=2):
    """Separa apenas train/good; test permanece fora da seleção aleatória.

    A fração é de componentes/grupos quando existirem relações. O tamanho real
    por imagem é registrado. Sem grupos, cada imagem é um componente.
    """
    if isinstance(calibration_fraction, bool) or not isinstance(calibration_fraction, Real) or not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction deve estar entre 0 e 1.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed deve ser inteiro entre 0 e 2**32-1.")
    _positivo(min_fit, "min_fit")
    _positivo(min_calibration, "min_calibration")
    if min_calibration < 2:
        raise ValueError("Reserve pelo menos duas amostras para calibração.")
    frame, grupos = _catalogo(df, group_columns)
    components = _componentes(frame, grupos)
    train_groups = [g for g in components if frame.iloc[g[0]]["split"] == "train"]
    if len(train_groups) < 2:
        raise ValueError("São necessários pelo menos dois grupos independentes de treino.")
    count = int(np.ceil(len(train_groups) * calibration_fraction))
    order = np.random.default_rng(seed).permutation(len(train_groups))
    cal_rows = sorted(i for j in order[:count] for i in train_groups[j])
    fit_rows = sorted(i for j in order[count:] for i in train_groups[j])
    if len(fit_rows) < min_fit or len(cal_rows) < min_calibration:
        raise ValueError("Divisão pequena: aumente o dataset ou ajuste a proporção respeitando min_fit/min_calibration e os grupos.")
    manifest = {
        "version": 1, "seed": int(seed), "calibration_fraction": float(calibration_fraction),
        "group_columns": grupos, "min_fit": int(min_fit), "min_calibration": int(min_calibration),
        "dataset_fingerprint": _digest(frame.to_dict("records")),
        "uses_content_hashes": "sha256" in frame,
        "fit_ids": frame.iloc[fit_rows]["caminho"].tolist(),
        "calibration_ids": frame.iloc[cal_rows]["caminho"].tolist(),
        "test_ids": frame.loc[frame["split"] == "test", "caminho"].tolist(),
        "realized_calibration_fraction": len(cal_rows) / (len(cal_rows) + len(fit_rows)),
    }
    manifest["manifest_hash"] = _digest(manifest)
    return manifest


def _verificar_integridade(manifest):
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Versão de manifesto inválida.")
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if manifest.get("manifest_hash") != _digest(body):
        raise ValueError("Manifesto alterado: hash de integridade não confere.")


def validar_divisao(manifest, df):
    _verificar_integridade(manifest)
    frame, grupos = _catalogo(df, manifest["group_columns"])
    if manifest["dataset_fingerprint"] != _digest(frame.to_dict("records")):
        raise ValueError("Inventário mudou: caminhos, rótulos, grupos ou hashes diferem do manifesto.")
    partitions = [manifest[k] for k in ("fit_ids", "calibration_ids", "test_ids")]
    if any(len(ids) != len(set(ids)) for ids in partitions):
        raise ValueError("IDs repetidos no manifesto.")
    fit, cal, test = map(set, partitions)
    train_ids = set(frame.loc[frame["split"] == "train", "caminho"])
    test_ids = set(frame.loc[frame["split"] == "test", "caminho"])
    if fit & cal or (fit | cal) & test or (fit | cal) != train_ids or test != test_ids:
        raise ValueError("Partições sobrepostas, incompletas ou contendo test no treino.")
    if len(fit) < manifest["min_fit"] or len(cal) < manifest["min_calibration"]:
        raise ValueError("Partições menores que os mínimos registrados.")
    for members in _componentes(frame, grupos):
        ids = set(frame.iloc[members]["caminho"])
        if ids & fit and ids & cal:
            raise ValueError("Grupo/hash dividido entre ajuste e calibração.")
    return frame


def salvar_divisao(manifest, caminho):
    """Criação exclusiva: nunca sobrescreve um manifesto de outro experimento."""
    _verificar_integridade(manifest)
    text = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(caminho)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def carregar_divisao(caminho, df):
    with Path(caminho).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validar_divisao(manifest, df)
    return manifest


def obter_ou_criar_divisao(df, caminho, **config):
    """Reusa somente a mesma divisão/configuração; mudanças exigem novo destino."""
    proposta = gerar_divisao(df, **config)
    if Path(caminho).exists():
        existente = carregar_divisao(caminho, df)
        if existente != proposta:
            raise ValueError("Configuração/divisão diferente do manifesto existente; use outro arquivo explicitamente.")
        return existente
    salvar_divisao(proposta, caminho)
    return proposta


@dataclass(frozen=True)
class CaracteristicasPreparadas:
    preprocessor: Pipeline
    X_fit: np.ndarray
    X_calibration: np.ndarray
    fit_ids: tuple
    calibration_ids: tuple
    feature_columns: tuple
    manifest_hash: str


def _matriz(frame, ids, columns):
    return check_array(frame.loc[list(ids), list(columns)].to_numpy(), dtype=np.float64)


def preparar_caracteristicas(df, manifest, feature_columns, *, selector=None, pca=None):
    """Ajusta seletores não supervisionados, scaler e PCA SOMENTE em fit_ids.

    Clona transformadores fornecidos, descartando eventual estado já ajustado.
    Não lê valores das características de test para ajuste ou calibração.
    """
    validar_divisao(manifest, df)
    columns = tuple(feature_columns)
    if not columns or len(set(columns)) != len(columns) or any(
        not isinstance(c, str) or re.fullmatch(r"feat_\d+", c) is None or c not in df for c in columns
    ):
        raise ValueError("Selecione explicitamente colunas feat_N únicas e existentes.")
    frame = normalizar_caminhos(df).set_index("caminho")
    X_fit = _matriz(frame, manifest["fit_ids"], columns)
    X_cal = _matriz(frame, manifest["calibration_ids"], columns)
    steps = []
    if selector is not None:
        steps.append(("selector", clone(selector)))
    steps.append(("scaler", StandardScaler()))
    if pca is not None:
        steps.append(("pca", clone(pca)))
    preprocessor = Pipeline(steps)
    X_fit = preprocessor.fit_transform(X_fit)
    X_cal = preprocessor.transform(X_cal)
    return CaracteristicasPreparadas(
        preprocessor, check_array(X_fit), check_array(X_cal),
        tuple(manifest["fit_ids"]), tuple(manifest["calibration_ids"]), columns, manifest["manifest_hash"],
    )


def transformar_teste(df, manifest, prepared):
    """Aplicação futura no teste: transform apenas, na ordem test_ids."""
    validar_divisao(manifest, df)
    if prepared.manifest_hash != manifest["manifest_hash"]:
        raise ValueError("Transformador pertence a outra divisão.")
    if not manifest["test_ids"]:
        raise ValueError("Manifesto sem imagens de teste.")
    frame = normalizar_caminhos(df).set_index("caminho")
    return prepared.preprocessor.transform(_matriz(frame, manifest["test_ids"], prepared.feature_columns))
