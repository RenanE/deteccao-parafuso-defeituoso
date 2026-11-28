"""Etapas 4/5: comparação interna congelada; nunca abre imagens das reservas."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .feature_experiments import (
    ROOT, environment, file_hash, load_features, load_study,
    reserve_directory, save_json, signature,
)
from .features import HOG_CONFIGS
from .knn import KNNAnomalyDetector
from .ocsvm import CalibratedOneClassSVM
from .mahalanobis import MahalanobisAnomalyDetector, NumericalFitError
from .isolation_forest import IsolationForestAnomalyDetector
from .representations import FeatureRepresentation


REPRESENTATIONS = {
    "R1": {"hog": "A", "scope": "none", "components": None, "dimension": 25413},
    "R2": {"hog": "C", "scope": "hog", "components": 10, "dimension": 43},
    "R3": {"hog": "C", "scope": "full", "components": 25, "dimension": 25},
}
TARGET_FPR = 0.05
SELECTION_RULE = (
    "Minimizar RMSE da rejeição de boas versus 0.05 nas três repetições; "
    "empates em 12 casas: menor dimensão, depois configuração em ordem lexical. "
    "Critério exploratório, sem inferência sobre recall de defeitos."
)


def protocol_sources():
    names = ("anomaly.py", "knn.py", "ocsvm.py", "representations.py", "validation.py",
             "feature_experiments.py", "model_experiments.py", "final_calibration.py",
             "mahalanobis.py", "isolation_forest.py", "selection_bundle.py")
    return {name: file_hash(ROOT / "src" / name) for name in names}


def parameter_grid(stage=4):
    if stage == 5:
        return ([{"model": "mahalanobis", "parameters": {"covariance": method,
                  "support_fraction": support, "random_state": 42}}
                 for method, support in (("ledoit_wolf", None), ("elliptic", None), ("elliptic", .75))]
                + [{"model": "isolation_forest", "parameters": {"n_estimators": trees,
                    "max_samples": samples, "max_features": features, "random_state": 42}}
                   for trees in (100, 200) for samples in (64, 128) for features in (.5, 1.0)])
    if stage != 4:
        raise ValueError("stage deve ser 4 ou 5.")
    return ([{"model": "svm", "parameters": {"nu": nu, "gamma_multiplier": gamma}}
             for nu in (0.01, 0.05, 0.10) for gamma in (0.5, 1.0, 2.0)]
            + [{"model": "knn", "parameters": {"n_neighbors": k, "aggregation": aggregation}}
               for k in (3, 5, 10, 20) for aggregation in ("mean", "kth")])


def make_detector(spec):
    cls = {"svm": CalibratedOneClassSVM, "knn": KNNAnomalyDetector,
           "mahalanobis": MahalanobisAnomalyDetector,
           "isolation_forest": IsolationForestAnomalyDetector}.get(spec["model"])
    if cls is None:
        raise ValueError("Modelo desconhecido.")
    return cls(target_fpr=TARGET_FPR, **spec["parameters"])


def make_representation(name, seed=42):
    spec = REPRESENTATIONS[name]
    return FeatureRepresentation(HOG_CONFIGS[spec["hog"]].hog_dimension,
                                 spec["scope"], spec["components"], seed=seed)


def load_development(study, cache, *, verify_images=True):
    """Valida metadados das reservas; abre bytes somente dos IDs de desenvolvimento."""
    inventory, outer, inner = load_study(study)
    if inner["seeds"] != [42, 43, 44]:
        raise ValueError("O protocolo exige as sementes internas congeladas 42/43/44.")
    matrices, metadata = {}, {}
    for name in ("A", "C"):
        X, ids, meta = load_features(cache, name, inventory, outer)
        if ids != outer["fit_ids"]:
            raise ValueError("Ordem dos IDs do cache difere do manifesto congelado.")
        matrices[name], metadata[name] = X, meta
    if verify_images:
        hashes = inventory.set_index("caminho")["sha256"]
        for path in outer["fit_ids"]:
            if file_hash(path) != hashes[path]:
                raise ValueError(f"Imagem de desenvolvimento alterada: {path}")
    return inventory, outer, inner, matrices, metadata


def resources():
    memory = None
    info = Path("/proc/meminfo")
    if info.exists():
        memory = int(next(line.split()[1] for line in info.read_text().splitlines()
                          if line.startswith("MemAvailable:"))) * 1024
    return {"available_memory_bytes": memory, "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "estimated_working_budget_bytes": 1024 ** 3, "estimated_disk_budget_bytes": 256 * 1024 ** 2}


def peak_process_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def score_summary(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Distribuição de scores inválida.")
    return {"n": len(scores), "mean": float(scores.mean()), "std": float(scores.std()),
            "min": float(scores.min()), "q05": float(np.quantile(scores, .05)),
            "q25": float(np.quantile(scores, .25)), "median": float(np.median(scores)),
            "q75": float(np.quantile(scores, .75)), "q95": float(np.quantile(scores, .95)),
            "max": float(scores.max()), "unique": len(np.unique(scores))}


def model_array_bytes(detector):
    """Arrays explícitos únicos, incluindo referências k-NN; não é pico RSS."""
    seen = set()
    def count(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if hasattr(obj, "node_count") and hasattr(obj, "__getstate__"):
            return sum(v.nbytes for v in obj.__getstate__().values() if isinstance(v, np.ndarray))
        if isinstance(obj, (list, tuple)):
            return sum(count(v) for v in obj)
        if isinstance(obj, dict):
            return sum(count(v) for v in obj.values())
        if hasattr(obj, "__dict__"):
            return sum(count(v) for v in vars(obj).values()
                       if isinstance(v, (np.ndarray, list, tuple, dict))
                       or hasattr(v, "get_params") or hasattr(v, "node_count"))
        return 0
    return count(detector)


def summarize(rows):
    groups = {}
    for row in rows:
        key = json.dumps([row["representation"], row["model"], row["parameters"]], sort_keys=True)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, values in groups.items():
        rates = np.array([v["normal_rejection_rate"] for v in values])
        item = {k: values[0][k] for k in ("representation", "model", "parameters", "output_dimension")}
        item.update(config_key=key, repeats=len(values), seeds=[v["seed"] for v in values],
                    mean_rejection=float(rates.mean()), std_rejection=float(rates.std()),
                    min_rejection=float(rates.min()), max_rejection=float(rates.max()),
                    target_rmse=float(np.sqrt(np.mean((rates - TARGET_FPR) ** 2))),
                    rejections=[v["normal_rejections"] for v in values])
        for name in ("transform_fit_seconds", "transform_apply_seconds", "model_fit_seconds",
                     "calibration_seconds", "validation_score_seconds", "model_array_bytes",
                     "transformer_array_bytes", "transformed_array_bytes"):
            item[name + "_mean"] = float(np.mean([v[name] for v in values]))
        summaries.append(item)
    return summaries


def choose_provisional(summaries, models=("svm", "knn")):
    if any(s["seeds"] != [42, 43, 44] for s in summaries):
        raise ValueError("Seleção exige três repetições completas; piloto não seleciona modelos.")
    def key(s):
        return (round(s["target_rmse"], 12), s["output_dimension"], s["config_key"])
    return {model: min((s for s in summaries if s["model"] == model), key=key)
            for model in models}


def compare_models(study, cache, out, *, pilot=False, stage=4):
    if Path(out).exists() or Path(out).is_symlink():
        raise FileExistsError(f"Destino existente: {out}")
    available = resources()
    if available["disk_free_bytes"] < available["estimated_disk_budget_bytes"]:
        raise OSError("Espaço em disco insuficiente para a margem de execução.")
    if available["available_memory_bytes"] is not None and available["available_memory_bytes"] < available["estimated_working_budget_bytes"]:
        raise MemoryError("Memória disponível abaixo da margem de 1 GiB; execução interrompida.")
    _, outer, inner, matrices, metadata = load_development(study, cache)
    complete_grid = parameter_grid() if stage == 4 else parameter_grid(stage)
    models = ("svm", "knn") if stage == 4 else ("mahalanobis", "isolation_forest")
    protocol = {"stage": stage, "outer_manifest_hash": outer["manifest_hash"], "inner_signature": inner["signature"],
                "representations": REPRESENTATIONS, "grid": complete_grid, "target_fpr": TARGET_FPR,
                "selection_rule": SELECTION_RULE, "sources": protocol_sources(),
                "cache_hashes": {k: m["cache_sha256"] for k, m in metadata.items()},
                "seeds": [42, 43, 44], "pilot": pilot, "resources_before": available,
                "environment": environment(), "threads": 1}
    dest = reserve_directory(out)
    save_json(dest / "protocol.json", protocol)
    rows, skipped = [], []
    start_total = time.perf_counter()
    positions = {p: i for i, p in enumerate(outer["fit_ids"])}
    grid = complete_grid if not pilot else [
        {"model": "svm", "parameters": {"nu": .05, "gamma_multiplier": 1.0}},
        {"model": "knn", "parameters": {"n_neighbors": 5, "aggregation": "mean"}}]
    if stage == 5 and pilot:
        grid = [s for s in complete_grid if s["model"] == "mahalanobis"] + [
            next(s for s in complete_grid if s["model"] == "isolation_forest")]
    representations = ["R2"] if pilot and stage == 4 else REPRESENTATIONS
    try:
        with threadpool_limits(limits=1), (dest / "runs.jsonl").open("x") as log:
            for name in representations:
                for fold in (inner["folds"][:1] if pilot else inner["folds"]):
                    X = matrices[REPRESENTATIONS[name]["hog"]]
                    raw = [X[[positions[p] for p in fold[key]]] for key in ("fit_ids", "threshold_ids", "validation_ids")]
                    transform = make_representation(name, seed=fold["seed"])
                    start = time.perf_counter()
                    transform.fit(raw[0])
                    fit_time = time.perf_counter() - start
                    start = time.perf_counter()
                    fit, calibration, validation = [transform.transform(x) for x in raw]
                    apply_time = time.perf_counter() - start
                    if fit.shape[1] != REPRESENTATIONS[name]["dimension"]:
                        raise ValueError("Dimensão da representação incorreta.")
                    for spec in grid:
                        if spec["model"] == "mahalanobis" and name == "R1":
                            continue  # Nunca alocar covariância de 25.413 dimensões.
                        model = make_detector(spec)
                        start = time.perf_counter()
                        try:
                            model.fit(fit, sample_ids=fold["fit_ids"])
                        except NumericalFitError as exc:
                            failed = {"status": "skipped", "representation": name, **spec,
                                      "seed": fold["seed"], "output_dimension": fit.shape[1],
                                      "reason": str(exc), "fit_seconds": time.perf_counter() - start}
                            skipped.append(failed)
                            log.write(json.dumps(failed, allow_nan=False) + "\n")
                            log.flush()
                            continue
                        model_time = time.perf_counter() - start
                        start = time.perf_counter()
                        model.calibrate(calibration, sample_ids=fold["threshold_ids"])
                        calibration_time = time.perf_counter() - start
                        start = time.perf_counter()
                        scores = model.anomaly_score(validation)
                        predictions = np.where(scores > model.threshold_, -1, 1)
                        score_time = time.perf_counter() - start
                        reference_scores = (model.training_scores_ if spec["model"] == "knn"
                                            else model.anomaly_score(fit))
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
                               "scores": {"fit_reference": score_summary(reference_scores),
                                          "calibration": score_summary(model.calibration_scores_),
                                          "validation": score_summary(scores)},
                               "reference_score_note": "k-NN: leave-self-out; demais: in-sample",
                               "calibration_ids": fold["threshold_ids"], "validation_ids": fold["validation_ids"],
                               "calibration_scores": model.calibration_scores_.tolist(),
                               "validation_scores": scores.tolist(), "validation_predictions": predictions.tolist(),
                               "validation_threshold_ties": int(np.sum(scores == model.threshold_))}
                        if hasattr(model, "diagnostics_"):
                            row["numerical_diagnostics"] = model.diagnostics_
                        if spec["model"] == "svm":
                            row.update(resolved_gamma=model.resolved_gamma_, n_support=len(model.model_.support_),
                                       native_svm_rejection=float(np.mean(model.model_.predict(validation) == -1)),
                                       historical_hyperparameters=(spec["parameters"] == {"nu": .05, "gamma_multiplier": 1.0}))
                        rows.append(row)
                        log.write(json.dumps(row, allow_nan=False) + "\n")
                        log.flush()
                    print(f"{name}, semente {fold['seed']}: acumulado {len(rows)} válidas, {len(skipped)} inviáveis", flush=True)
        summaries = summarize(rows)
        report = {"status": "complete", "stage": stage, "pilot": pilot, "protocol_signature": signature(protocol),
                  "rows": rows, "skipped": skipped, "summary": summaries, "selection_rule": SELECTION_RULE,
                  "wall_seconds": time.perf_counter() - start_total,
                  "peak_process_rss_bytes": peak_process_rss_bytes(),
                  "memory_note": "Arrays retidos aproximados por configuração; RSS máximo do processo inteiro, Linux.",
                  "limitations": "Somente normais; três repetições correlacionadas; seleção pós-etapa 3, sem estimativa imparcial de recall."}
        save_json(dest / "comparison.json", report)
        with (dest / "summary.csv").open("x", newline="") as stream:
            pd.DataFrame(summaries).to_csv(stream, index=False)
        if not pilot:
            eligible = [s for s in summaries if s["seeds"] == [42, 43, 44]]
            selection = {"selected": choose_provisional(eligible, models=models), "selection_rule": SELECTION_RULE,
                         "outer_manifest_hash": outer["manifest_hash"], "inner_signature": inner["signature"],
                         "sources": protocol_sources(), "cache_hashes": protocol["cache_hashes"],
                         "representations": REPRESENTATIONS, "target_fpr": TARGET_FPR,
                         "comparison_sha256": file_hash(dest / "comparison.json"),
                         "final_calibration_executed": False}
            selection["signature"] = signature(selection)
            save_json(dest / "selection.json", selection)
        print(f"Etapa {stage} salva em {dest}: {len(rows)} válidas, {len(skipped)} inviáveis; calibração final não utilizada.")
        return report
    except Exception as exc:
        save_json(dest / "failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                                         "completed_runs": len(rows), "protocol_changed": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Somente semente 42; etapa 4: dois modelos/R2; etapa 5: grade reduzida/R1-R3. Sem seleção.")
    parser.add_argument("--stage", type=int, choices=(4, 5), default=4)
    args = parser.parse_args(argv)
    try:
        compare_models(args.study, args.cache, args.out, pilot=args.pilot, stage=args.stage)
    except (ValueError, OSError, MemoryError) as exc:
        parser.exit(1, f"Etapa {args.stage} interrompida: {exc}\n")


if __name__ == "__main__":
    main()
