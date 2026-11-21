"""Etapa 3: preparar inventário, extrair apenas ajuste e comparar normais internos.

Nenhum comando extrai características de test/ ou da calibração final.
Saídas são criadas em diretórios novos; não há modo de sobrescrita.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .dataset import adicionar_argumento_confirmacao, carregar_caminhos_dataset, resolver_data_dir
from .extract_features import processar_imagem
from .features import HOG_CONFIGS, criar_banco_gabor
from .knn import KNNAnomalyDetector
from .representations import FeatureRepresentation, InvalidPCADimension
from .validation import (
    adicionar_hashes, gerar_divisao, normalizar_caminhos, salvar_divisao, validar_divisao,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = ("features.py", "preprocessing.py", "segmentation.py", "extract_features.py")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def extraction_sources():
    return {name: file_hash(ROOT / "src" / name) for name in SOURCE_FILES}


def environment():
    packages = ("numpy", "scipy", "scikit-image", "opencv-python", "scikit-learn", "pandas", "threadpoolctl")
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "extraction_sources": extraction_sources(),
        "experiment_source": file_hash(__file__),
        "representation_source": file_hash(ROOT / "src/representations.py"),
    }


def reserve_directory(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def audit_dataset(data_dir, inventory):
    """Inspeção de nomes e chunks PNG; não inventa IDs com base em sequências."""
    treino = inventory.loc[inventory.split == "train", "caminho"]
    metadata = {}
    for filename in treino:
        keys = []
        with Path(filename).open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Arquivo PNG com assinatura inesperada: {filename}")
            while header := stream.read(8):
                if len(header) != 8:
                    raise ValueError(f"Cabeçalho PNG incompleto: {filename}")
                size, kind = struct.unpack(">I4s", header)
                if kind in (b"tEXt", b"zTXt", b"iTXt"):
                    data = stream.read(min(size, 1024))
                    keys.append(kind.decode() + ":" + data.split(b"\0", 1)[0].decode("latin1"))
                    stream.seek(size - len(data) + 4, 1)
                else:
                    if kind == b"eXIf":
                        keys.append("eXIf (requer interpretação manual)")
                    stream.seek(size + 4, 1)
                if kind == b"IEND":
                    break
        if keys:
            metadata[filename] = keys
    old_csvs = []
    seen = set()
    for base in (Path(data_dir), ROOT / "data"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.csv")):
            if str(path.resolve()) in seen:
                continue
            seen.add(str(path.resolve()))
            item = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_hash(path),
                    "configuration": "Não verificável apenas pelo CSV; não reutilizado."}
            try:
                cols = pd.read_csv(path, nrows=0).columns
                item["feature_columns"] = sum(col.startswith("feat_") for col in cols)
            except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
                item["header_error"] = str(exc)
            old_csvs.append(item)
    auxiliary = [str(p) for p in sorted(Path(data_dir).rglob("*"))
                 if p.is_file() and p.suffix.lower() != ".png"]
    return {
        "train_filename_examples": [Path(p).name for p in treino.iloc[:12]],
        "all_train_stems_numeric": all(Path(p).stem.isdigit() for p in treino),
        "png_metadata_keys": metadata, "auxiliary_files": auxiliary,
        "group_inference": "Nenhum grupo inferido de nomes ou metadados; fornecer IDs explícitos se disponíveis.",
        "historical_csvs": old_csvs,
        "normal_labels": "good conforme estrutura/declaracão explícita exigida pelo carregador; a fonte humana deve ser registrada no procedimento.",
    }


def internal_splits(inventory, outer, *, seeds=(42, 43, 44), validation_fraction=0.2, threshold_fraction=0.2):
    validar_divisao(outer, inventory)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Forneça sementes internas únicas.")
    pool = inventory[inventory.caminho.isin(outer["fit_ids"])].copy()
    folds = []
    for seed in seeds:
        first = gerar_divisao(pool, seed=seed, calibration_fraction=validation_fraction,
                             group_columns=outer["group_columns"], min_fit=8)
        remaining = pool[pool.caminho.isin(first["fit_ids"])]
        second = gerar_divisao(remaining, seed=seed + 1000, calibration_fraction=threshold_fraction,
                              group_columns=outer["group_columns"])
        folds.append({"seed": seed, "fit_ids": second["fit_ids"],
                      "threshold_ids": second["calibration_ids"], "validation_ids": first["calibration_ids"]})
    result = {"version": 1, "outer_manifest_hash": outer["manifest_hash"],
              "seeds": list(seeds), "validation_fraction": validation_fraction,
              "threshold_fraction": threshold_fraction, "folds": folds}
    result["signature"] = signature(result)
    return result


def prepare_study(data_dir, out_dir, *, train_flat_is_good=None, groups_csv=None):
    if Path(out_dir).exists():
        raise FileExistsError(f"Diretório já existente: {out_dir}")
    data_dir = resolver_data_dir(data_dir).resolve()
    inventory = adicionar_hashes(carregar_caminhos_dataset(data_dir, train_flat_is_good=train_flat_is_good))
    audit = audit_dataset(data_dir, inventory)
    if groups_csv is not None:
        groups = normalizar_caminhos(pd.read_csv(groups_csv), base_dir=data_dir)
        columns = [c for c in groups if c != "caminho"]
        if not columns or set(columns) & set(inventory.columns):
            raise ValueError("CSV de grupos deve conter caminho e apenas identificadores novos.")
        if set(groups.caminho) != set(inventory.caminho):
            raise ValueError("CSV de grupos deve identificar todas as imagens do inventário.")
        inventory = inventory.merge(groups, on="caminho", validate="one_to_one")
        audit["groups_file"] = {"path": str(Path(groups_csv).resolve()), "sha256": file_hash(groups_csv)}
    else:
        columns = None
    outer = gerar_divisao(inventory, seed=42, calibration_fraction=0.2, group_columns=columns)
    inner = internal_splits(inventory, outer)
    audit["group_columns"] = outer["group_columns"]
    audit["counts"] = inventory.groupby(["split", "classe"]).size().reset_index(name="count").to_dict("records")
    out = reserve_directory(out_dir)
    save_json(out / "inventory.json", inventory.to_dict("records"))
    salvar_divisao(outer, out / "outer_manifest.json")
    save_json(out / "internal_manifest.json", inner)
    save_json(out / "audit.json", audit)
    save_json(out / "environment.json", environment())
    print(f"Estudo salvo em {out}: ajuste={len(outer['fit_ids'])}, calibração final={len(outer['calibration_ids'])}, teste={len(outer['test_ids'])}")
    return out


def load_study(study_dir):
    root = Path(study_dir)
    def read(name):
        return json.loads((root / name).read_text())
    inventory = pd.DataFrame(read("inventory.json"))
    outer, inner = read("outer_manifest.json"), read("internal_manifest.json")
    validar_divisao(outer, inventory)
    expected = internal_splits(inventory, outer, seeds=inner["seeds"],
                               validation_fraction=inner["validation_fraction"], threshold_fraction=inner["threshold_fraction"])
    if expected != inner:
        raise ValueError("Manifesto interno alterado ou incompatível com a divisão externa.")
    return inventory, outer, inner


def extract_study(study_dir, out_dir, *, configs=("A", "B", "C", "D"), sample_size=8, full=False):
    if Path(out_dir).exists():
        raise FileExistsError(f"Diretório já existente: {out_dir}")
    if not configs or len(set(configs)) != len(configs) or any(c not in HOG_CONFIGS for c in configs):
        raise ValueError("Configurações devem ser uma seleção única de A, B, C, D.")
    inventory, outer, _ = load_study(study_dir)
    ids = list(outer["fit_ids"])
    if not full:
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 1 <= sample_size <= len(ids):
            raise ValueError("sample_size deve estar entre 1 e o tamanho do ajuste.")
        ids = sorted(np.random.default_rng(42).choice(ids, size=sample_size, replace=False).tolist())
    by_id = inventory.set_index("caminho")
    # Somente hashes/imagens dos IDs de ajuste selecionados são lidos nesta fase.
    for path in ids:
        if file_hash(path) != by_id.loc[path, "sha256"]:
            raise ValueError(f"Imagem mudou desde o inventário: {path}")
    out = reserve_directory(out_dir)
    bank = criar_banco_gabor(orientations=4, lambdas=[8.0, 12.0, 16.0])
    for name in configs:
        config = HOG_CONFIGS[name]
        matrix = np.empty((len(ids), config.total_dimension), dtype=np.float32)
        durations = []
        for i, path in enumerate(ids):
            start = time.perf_counter()
            vector, ok = processar_imagem(Path(path), bank, hog_config=config)
            durations.append(time.perf_counter() - start)
            if not ok or vector is None or vector.shape != (config.total_dimension,) or not np.isfinite(vector).all():
                raise ValueError(f"Características inválidas em {path}; extração interrompida.")
            matrix[i] = vector
        cache = out / f"features_{name}.npz"
        with cache.open("xb") as stream:
            np.savez_compressed(stream, features=matrix, ids=np.asarray(ids),
                                image_hashes=np.asarray(by_id.loc[ids, "sha256"].tolist()))
        meta = {
            "config_name": name, "config": config.metadata(), "seed": 42,
            "outer_manifest_hash": outer["manifest_hash"], "scope": "outer_fit_only",
            "complete_outer_fit": full, "sample_count": len(ids), "dtype": str(matrix.dtype),
            "extraction_seconds": sum(durations), "seconds_per_image": durations,
            "array_bytes": matrix.nbytes, "disk_bytes": cache.stat().st_size,
            "cache_sha256": file_hash(cache), "environment": environment(),
        }
        save_json(out / f"features_{name}.json", meta)
        print(f"{name}: {matrix.shape}, {sum(durations):.3f}s, arrays={matrix.nbytes} bytes", flush=True)
    save_json(out / "run.json", {"status": "complete", "configs": list(configs), "seed": 42,
                                  "full": full, "sample_count": len(ids), "outer_manifest_hash": outer["manifest_hash"]})
    return out


def load_features(cache_dir, name, inventory, outer):
    root = Path(cache_dir)
    run = json.loads((root / "run.json").read_text())
    meta = json.loads((root / f"features_{name}.json").read_text())
    path = root / f"features_{name}.npz"
    if run.get("status") != "complete" or not meta["complete_outer_fit"]:
        raise ValueError("Comparação completa exige cache completo do ajuste; amostra piloto não é suficiente.")
    if meta["outer_manifest_hash"] != outer["manifest_hash"] or meta["config"] != HOG_CONFIGS[name].metadata():
        raise ValueError("Cache pertence a outra divisão/configuração.")
    if meta["environment"]["extraction_sources"] != extraction_sources():
        raise ValueError("Código de extração mudou; use um novo cache versionado.")
    if file_hash(path) != meta["cache_sha256"]:
        raise ValueError("Hash do cache não confere.")
    with np.load(path, allow_pickle=False) as arrays:
        X, ids, hashes = arrays["features"], arrays["ids"].tolist(), arrays["image_hashes"].tolist()
    if len(set(ids)) != len(ids) or set(ids) != set(outer["fit_ids"]):
        raise ValueError("Cache inclui IDs indevidos ou omite parte do ajuste; teste/calibração final são proibidos.")
    if hashes != inventory.set_index("caminho").loc[ids, "sha256"].tolist():
        raise ValueError("Hashes das imagens no cache não conferem com o inventário.")
    if X.shape != (len(ids), HOG_CONFIGS[name].total_dimension) or not np.isfinite(X).all():
        raise ValueError("Matriz de características inválida.")
    return X, ids, meta


def compare_study(study_dir, cache_dir, out_dir, *, configs=("A", "B", "C", "D"),
                  components=(10, 25, 50, 100), n_neighbors=5, target_fpr=0.05, threads=1):
    if Path(out_dir).exists():
        raise FileExistsError(f"Diretório já existente: {out_dir}")
    if not configs or len(set(configs)) != len(configs) or any(c not in HOG_CONFIGS for c in configs):
        raise ValueError("Configurações inválidas.")
    if not components or len(set(components)) != len(components) or any(type(n) is not int or n < 1 for n in components):
        raise ValueError("Componentes devem ser inteiros positivos únicos.")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads deve ser positivo.")
    inventory, outer, inner = load_study(study_dir)
    out = reserve_directory(out_dir)
    rows = []
    variants = [("none", None)] + [(scope, n) for scope in ("full", "hog") for n in components]
    with threadpool_limits(limits=threads):
        for name in configs:
            X, ids, extraction = load_features(cache_dir, name, inventory, outer)
            positions = {sample: i for i, sample in enumerate(ids)}
            for fold in inner["folds"]:
                raw = [X[[positions[p] for p in fold[key]]] for key in ("fit_ids", "threshold_ids", "validation_ids")]
                for scope, n in variants:
                    row = {"hog_config": name, "pca_scope": scope, "n_components": n, "seed": fold["seed"],
                           "input_dimension": X.shape[1], "n_fit": len(raw[0]), "n_threshold": len(raw[1]),
                           "n_validation": len(raw[2]), "target_fpr": target_fpr, "n_neighbors": n_neighbors,
                           "extraction_seconds_total": extraction["extraction_seconds"], "cache_array_bytes": X.nbytes}
                    rep = FeatureRepresentation(HOG_CONFIGS[name].hog_dimension, scope, n, seed=fold["seed"])
                    start = time.perf_counter()
                    try:
                        rep.fit(raw[0])
                    except InvalidPCADimension as exc:
                        rows.append({**row, "status": "skipped", "reason": str(exc)})
                        continue
                    row["transform_fit_seconds"] = time.perf_counter() - start
                    start = time.perf_counter()
                    fit, threshold, validation = [rep.transform(block) for block in raw]
                    row["transform_apply_seconds"] = time.perf_counter() - start
                    detector = KNNAnomalyDetector(n_neighbors=n_neighbors, target_fpr=target_fpr)
                    start = time.perf_counter()
                    detector.fit(fit, sample_ids=fold["fit_ids"])
                    detector.calibrate(threshold, sample_ids=fold["threshold_ids"])
                    row["knn_fit_calibrate_seconds"] = time.perf_counter() - start
                    start = time.perf_counter()
                    rejected = int(np.sum(detector.predict(validation) == -1))
                    row.update({"status": "ok", "score_seconds": time.perf_counter() - start,
                                "output_dimension": fit.shape[1], "effective_rank": rep.effective_rank_,
                                "explained_variance": rep.explained_variance_, "threshold": detector.threshold_,
                                "calibration_fpr": detector.calibration_fpr_, "normal_rejections": rejected,
                                "normal_rejection_rate": rejected / len(validation),
                                "raw_fold_array_bytes": sum(block.nbytes for block in raw),
                                "transformed_array_bytes": fit.nbytes + threshold.nbytes + validation.nbytes,
                                "transformer_state_bytes": rep.state_nbytes()})
                    rows.append(row)
    summary = []
    for name in configs:
        for scope, n in variants:
            selected = [r for r in rows if r["hog_config"] == name and r["pca_scope"] == scope and r["n_components"] == n and r["status"] == "ok"]
            if selected:
                rates = [r["normal_rejection_rate"] for r in selected]
                summary.append({"hog_config": name, "pca_scope": scope, "n_components": n,
                                "completed_repeats": len(selected), "normal_rejection_mean": float(np.mean(rates)),
                                "normal_rejection_std": float(np.std(rates)),
                                "transform_fit_seconds_mean": float(np.mean([r["transform_fit_seconds"] for r in selected]))})
    report = {"status": "complete", "outer_manifest_hash": outer["manifest_hash"], "inner_signature": inner["signature"],
              "environment": environment(), "threads": threads, "rows": rows, "summary": summary,
              "winner": None, "scope": "Somente normais do ajuste externo; sem teste ou calibração final.",
              "memory_note": "Bytes de arrays explícitos/estado; não é RSS nem pico de memória de BLAS/SVD.",
              "interpretation": "Rejeição de boas e estabilidade não medem recall de defeitos. Não selecionar vencedor apenas por variância ou dimensão."}
    save_json(out / "comparison.json", report)
    with (out / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(rows).to_csv(stream, index=False)
    print(f"Comparação interna salva em {out}; {sum(r['status'] == 'ok' for r in rows)} execuções válidas; sem vencedor automático.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Inventário/hashes e divisões, sem extração.")
    prepare.add_argument("--data-dir", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--groups-csv", type=Path)
    adicionar_argumento_confirmacao(prepare)
    extract = sub.add_parser("extract", help="Amostra piloto por padrão; --full extrai só os 256 IDs de ajuste.")
    extract.add_argument("--study", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--configs", nargs="+", choices=list(HOG_CONFIGS), default=list(HOG_CONFIGS))
    mode = extract.add_mutually_exclusive_group()
    mode.add_argument("--sample-size", type=int, default=8)
    mode.add_argument("--full", action="store_true")
    compare = sub.add_parser("compare", help="Validação interna apenas; sem avaliação final.")
    compare.add_argument("--study", type=Path, required=True)
    compare.add_argument("--cache", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--configs", nargs="+", choices=list(HOG_CONFIGS), default=list(HOG_CONFIGS))
    compare.add_argument("--components", nargs="+", type=int, default=[10, 25, 50, 100])
    compare.add_argument("--neighbors", type=int, default=5)
    compare.add_argument("--target-fpr", type=float, default=0.05)
    compare.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_study(args.data_dir, args.out, train_flat_is_good=args.train_flat_is_good, groups_csv=args.groups_csv)
        elif args.command == "extract":
            extract_study(args.study, args.out, configs=args.configs, sample_size=args.sample_size, full=args.full)
        else:
            compare_study(args.study, args.cache, args.out, configs=args.configs, components=args.components,
                          n_neighbors=args.neighbors, target_fpr=args.target_fpr, threads=args.threads)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Etapa 3 interrompida: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
