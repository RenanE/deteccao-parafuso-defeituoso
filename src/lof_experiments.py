"""Etapa 6: Local Outlier Factor no mesmo protocolo das etapas 4/5.

Subcomandos:
- `compare`: busca interna só nas 256 boas de desenvolvimento (sementes 42/43/44,
  163 ajuste / 41 limiar / 52 validação), mesma regra RMSE da etapa 4/5.
- `finalize`: ajusta a escolha nas 256 boas e calibra o limiar nas 64 boas
  reservadas (exige `--confirm-final-calibration`). Não lê o teste.

Os módulos já assinados (`model_experiments.py`, `final_calibration.py` etc.)
não são alterados; este módulo apenas reutiliza suas funções.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .extract_features import processar_imagem
from .feature_experiments import ROOT, environment, file_hash, reserve_directory, save_json, signature
from .features import HOG_CONFIGS, criar_banco_gabor
from .lof import LOFAnomalyDetector
from .model_experiments import (
    REPRESENTATIONS, SELECTION_RULE, TARGET_FPR, choose_provisional, load_development,
    make_representation, model_array_bytes, protocol_sources, resources, score_summary, summarize,
)

GRID = [{"model": "lof", "parameters": {"n_neighbors": k, "metric": "euclidean"}}
        for k in (5, 10, 20, 35)]


def lof_sources():
    sources = protocol_sources()
    for name in ("lof.py", "lof_experiments.py"):
        sources[name] = file_hash(ROOT / "src" / name)
    return sources


def compare(study, cache, out):
    if Path(out).exists() or Path(out).is_symlink():
        raise FileExistsError(f"Destino existente: {out}")
    _, outer, inner, matrices, metadata = load_development(study, cache)
    protocol = {"stage": 6, "outer_manifest_hash": outer["manifest_hash"], "inner_signature": inner["signature"],
                "representations": REPRESENTATIONS, "grid": GRID, "target_fpr": TARGET_FPR,
                "selection_rule": SELECTION_RULE, "sources": lof_sources(),
                "cache_hashes": {k: m["cache_sha256"] for k, m in metadata.items()},
                "seeds": [42, 43, 44], "resources_before": resources(), "environment": environment(),
                "threads": 1}
    dest = reserve_directory(out)
    save_json(dest / "protocol.json", protocol)
    positions = {p: i for i, p in enumerate(outer["fit_ids"])}
    rows = []
    start_total = time.perf_counter()
    with threadpool_limits(limits=1), (dest / "runs.jsonl").open("x") as log:
        for name in REPRESENTATIONS:
            for fold in inner["folds"]:
                X = matrices[REPRESENTATIONS[name]["hog"]]
                raw = [X[[positions[p] for p in fold[key]]] for key in ("fit_ids", "threshold_ids", "validation_ids")]
                transform = make_representation(name, seed=fold["seed"])
                t0 = time.perf_counter()
                transform.fit(raw[0])
                fit_time = time.perf_counter() - t0
                t0 = time.perf_counter()
                fit, calibration, validation = [transform.transform(x) for x in raw]
                apply_time = time.perf_counter() - t0
                for spec in GRID:
                    model = LOFAnomalyDetector(target_fpr=TARGET_FPR, **spec["parameters"])
                    t0 = time.perf_counter()
                    model.fit(fit, sample_ids=fold["fit_ids"])
                    model_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    model.calibrate(calibration, sample_ids=fold["threshold_ids"])
                    calibration_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    scores = model.anomaly_score(validation)
                    predictions = np.where(scores > model.threshold_, -1, 1)
                    score_time = time.perf_counter() - t0
                    row = {"representation": name, **spec, "seed": fold["seed"],
                           "output_dimension": fit.shape[1], "n_fit": len(fit),
                           "n_calibration": len(calibration), "n_validation": len(validation),
                           "threshold": model.threshold_, "calibration_fpr": model.calibration_fpr_,
                           "normal_rejections": int(np.sum(predictions == -1)),
                           "normal_rejection_rate": float(np.mean(predictions == -1)),
                           "transform_fit_seconds": fit_time, "transform_apply_seconds": apply_time,
                           "model_fit_seconds": model_time, "calibration_seconds": calibration_time,
                           "validation_score_seconds": score_time,
                           "transformer_array_bytes": transform.state_nbytes(),
                           "model_array_bytes": model_array_bytes(model),
                           "transformed_array_bytes": sum(x.nbytes for x in (fit, calibration, validation)),
                           "scores": {"fit_reference": score_summary(model.training_scores_),
                                      "calibration": score_summary(model.calibration_scores_),
                                      "validation": score_summary(scores)},
                           "calibration_ids": fold["threshold_ids"], "validation_ids": fold["validation_ids"],
                           "calibration_scores": model.calibration_scores_.tolist(),
                           "validation_scores": scores.tolist(),
                           "validation_predictions": predictions.tolist(),
                           "numerical_diagnostics": model.diagnostics_}
                    rows.append(row)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                print(f"{name}, semente {fold['seed']}: {len(rows)} execuções", flush=True)
    summaries = summarize(rows)
    report = {"status": "complete", "stage": 6, "protocol_signature": signature(protocol), "rows": rows,
              "skipped": [], "summary": summaries, "selection_rule": SELECTION_RULE,
              "wall_seconds": time.perf_counter() - start_total,
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "limitations": "Somente normais; três repetições correlacionadas; sem estimativa de recall."}
    save_json(dest / "comparison.json", report)
    with (dest / "summary.csv").open("x", newline="") as stream:
        pd.DataFrame(summaries).drop(columns=["seeds"]).to_csv(stream, index=False)
    selection = {"kind": "lof_v1", "selected": choose_provisional(summaries, models=("lof",)),
                 "selection_rule": SELECTION_RULE, "outer_manifest_hash": outer["manifest_hash"],
                 "inner_signature": inner["signature"], "sources": lof_sources(),
                 "cache_hashes": protocol["cache_hashes"], "representations": REPRESENTATIONS,
                 "target_fpr": TARGET_FPR, "comparison_sha256": file_hash(dest / "comparison.json"),
                 "final_calibration_executed": False}
    selection["signature"] = signature(selection)
    save_json(dest / "selection.json", selection)
    return dest


def finalize(study, cache, selection_path, out, *, confirm=False):
    if not confirm:
        raise ValueError("Calibração final exige --confirm-final-calibration.")
    if Path(out).exists() or Path(out).is_symlink():
        raise FileExistsError(f"Destino existente: {out}")
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text())
    if signature({k: v for k, v in selection.items() if k != "signature"}) != selection["signature"]:
        raise ValueError("Seleção alterada; assinatura inválida.")
    if selection["sources"] != lof_sources():
        raise ValueError("Código mudou desde a seleção do LOF.")
    if file_hash(selection_path.parent / "comparison.json") != selection["comparison_sha256"]:
        raise ValueError("Relatório da seleção alterado.")
    inventory, outer, inner, matrices, metadata = load_development(study, cache)
    if (selection["outer_manifest_hash"] != outer["manifest_hash"]
            or selection["cache_hashes"] != {k: v["cache_sha256"] for k, v in metadata.items()}):
        raise ValueError("Seleção incompatível com o estudo/cache.")
    choice = selection["selected"]["lof"]
    name = choice["representation"]
    config = HOG_CONFIGS[REPRESENTATIONS[name]["hog"]]
    hashes = inventory.set_index("caminho")["sha256"]
    ids = outer["calibration_ids"]
    for path in ids:
        if file_hash(path) != hashes[path]:
            raise ValueError(f"Imagem da calibração final alterada: {path}")
    dest = reserve_directory(out)
    with threadpool_limits(limits=1):
        bank = criar_banco_gabor(orientations=4, lambdas=[8.0, 12.0, 16.0])
        vectors = []
        for path in ids:
            vector, ok = processar_imagem(Path(path), bank, hog_config=config)
            if not ok or vector is None or not np.isfinite(vector).all():
                raise ValueError(f"Características inválidas na calibração: {path}")
            vectors.append(vector)
        raw = np.asarray(vectors, dtype=np.float32)
        representation = make_representation(name, seed=42)
        development = representation.fit_transform(matrices[REPRESENTATIONS[name]["hog"]])
        calibration = representation.transform(raw)
        detector = LOFAnomalyDetector(target_fpr=TARGET_FPR, **choice["parameters"])
        detector.fit(development, sample_ids=outer["fit_ids"])
        detector.calibrate(calibration, sample_ids=ids)
        bundle = {"representation": representation, "detector": detector, "selection": choice,
                  "outer_manifest_hash": outer["manifest_hash"]}
        with (dest / "model.joblib").open("xb") as stream:
            joblib.dump(bundle, stream)
        with (dest / "calibration_features.npz").open("xb") as stream:
            np.savez_compressed(stream, features=raw, ids=np.asarray(ids),
                                image_hashes=np.asarray([hashes[p] for p in ids]))
        save_json(dest / "final_calibration.json", {
            "status": "complete", "model": "lof", "selection": choice,
            "selection_sha256": file_hash(selection_path), "sources": lof_sources(),
            "outer_manifest_hash": outer["manifest_hash"], "n_fit": len(development),
            "n_calibration": len(calibration), "fit_ids": outer["fit_ids"], "calibration_ids": ids,
            "descriptor_config": config.metadata(), "seed": 42, "target_fpr": TARGET_FPR,
            "threshold": detector.threshold_, "calibration_fpr": detector.calibration_fpr_,
            "calibration_scores": detector.calibration_scores_.tolist(),
            "model_sha256": file_hash(dest / "model.joblib"),
            "calibration_cache_sha256": file_hash(dest / "calibration_features.npz"),
            "test_accessed": False})
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("compare", "finalize"):
        p = sub.add_parser(command)
        p.add_argument("--study", required=True, type=Path)
        p.add_argument("--cache", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        if command == "finalize":
            p.add_argument("--selection", required=True, type=Path)
            p.add_argument("--confirm-final-calibration", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            dest = compare(args.study, args.cache, args.out)
        else:
            dest = finalize(args.study, args.cache, args.selection, args.out,
                            confirm=args.confirm_final_calibration)
    except (ValueError, OSError, MemoryError) as exc:
        parser.exit(1, f"Etapa 6 (LOF) interrompida: {exc}\n")
    print(f"Saída: {dest}")


if __name__ == "__main__":
    main()
