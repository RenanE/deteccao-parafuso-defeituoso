"""Prepara seleção dos quatro modelos sem consumir calibração final ou teste.

Migração explícita da procedência da etapa 4: mantém os detectores e
transformações antigos intactos e reproduz suas seis execuções internas
selecionadas antes de aceitar a atualização da orquestração.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .feature_experiments import file_hash, reserve_directory, save_json, signature
from .model_experiments import (
    REPRESENTATIONS, TARGET_FPR, load_development, make_detector,
    make_representation, protocol_sources, resources,
)


def read_signed_selection(path):
    path = Path(path).resolve()
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection["signature"] != signature({k: v for k, v in selection.items() if k != "signature"}):
        raise ValueError("Assinatura da seleção inválida.")
    comparison = path.parent / "comparison.json"
    if file_hash(comparison) != selection["comparison_sha256"]:
        raise ValueError("Relatório de origem alterado.")
    return selection, json.loads(comparison.read_text(encoding="utf-8")), comparison


def prepare_bundle(study, cache, stage4, stage5, out):
    if Path(out).exists() or Path(out).is_symlink():
        raise FileExistsError(f"Destino existente: {out}")
    old, old_report, old_report_path = read_signed_selection(stage4)
    new, _, new_report_path = read_signed_selection(stage5)
    sources = protocol_sources()
    critical = ("anomaly.py", "knn.py", "ocsvm.py", "representations.py", "validation.py", "feature_experiments.py")
    if any(old["sources"].get(name) != sources[name] for name in critical):
        raise ValueError("Código metodológico da etapa 4 mudou; migração automática recusada.")
    changed = {name: {"before": digest, "after": sources.get(name)}
               for name, digest in old["sources"].items() if sources.get(name) != digest}
    if set(changed) - {"model_experiments.py", "final_calibration.py"}:
        raise ValueError("Mudanças não previstas na compatibilidade da etapa 4.")
    if new["sources"] != sources:
        raise ValueError("Código mudou desde a seleção da etapa 5.")
    available = resources()
    if (available["available_memory_bytes"] is not None
            and available["available_memory_bytes"] < available["estimated_working_budget_bytes"]):
        raise MemoryError("Memória insuficiente para verificação de compatibilidade.")
    if available["disk_free_bytes"] < available["estimated_disk_budget_bytes"]:
        raise OSError("Disco insuficiente para verificação de compatibilidade.")
    _, outer, inner, matrices, metadata = load_development(study, cache)
    for selection in (old, new):
        if (selection["outer_manifest_hash"] != outer["manifest_hash"]
                or selection["inner_signature"] != inner["signature"]
                or selection["representations"] != REPRESENTATIONS
                or selection["target_fpr"] != TARGET_FPR
                or selection["cache_hashes"] != {k: v["cache_sha256"] for k, v in metadata.items()}):
            raise ValueError("Seleções e estudo/cache/protocolo incompatíveis.")
    if set(old["selected"]) != {"svm", "knn"} or set(new["selected"]) != {"mahalanobis", "isolation_forest"}:
        raise ValueError("São necessárias seleções completas das etapas 4 e 5.")
    dest = reserve_directory(out)
    regressions = []
    positions = {p: i for i, p in enumerate(outer["fit_ids"])}
    try:
        with threadpool_limits(limits=1):
            for name, choice in old["selected"].items():
                records = [r for r in old_report["rows"] if (r["model"], r["representation"], r["parameters"])
                           == (name, choice["representation"], choice["parameters"])]
                if sorted(r["seed"] for r in records) != [42, 43, 44]:
                    raise ValueError("Referências selecionadas da etapa 4 incompletas.")
                for fold in inner["folds"]:
                    expected = next(r for r in records if r["seed"] == fold["seed"])
                    if expected["calibration_ids"] != fold["threshold_ids"] or expected["validation_ids"] != fold["validation_ids"]:
                        raise ValueError("IDs da referência antiga incompatíveis com o manifesto.")
                    X = matrices[REPRESENTATIONS[choice["representation"]]["hog"]]
                    raw = [X[[positions[p] for p in fold[key]]] for key in ("fit_ids", "threshold_ids", "validation_ids")]
                    transform = make_representation(choice["representation"], seed=fold["seed"]).fit(raw[0])
                    fit, cal, val = [transform.transform(x) for x in raw]
                    detector = make_detector(choice).fit(fit, sample_ids=fold["fit_ids"])
                    detector.calibrate(cal, sample_ids=fold["threshold_ids"])
                    scores = detector.anomaly_score(val)
                    np.testing.assert_allclose(scores, expected["validation_scores"], rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(detector.threshold_, expected["threshold"], rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal(detector.predict(val), expected["validation_predictions"])
                    regressions.append({"model": name, "seed": fold["seed"], "predictions_equal": True,
                                        "max_score_difference": float(np.max(np.abs(scores - expected["validation_scores"]))),
                                        "threshold_difference": float(abs(detector.threshold_ - expected["threshold"]))})
        compatibility = {"changed_orchestration_sources": changed, "unchanged_methodological_sources": list(critical),
                         "sources": sources, "regressions": regressions,
                         "rtol": 1e-12, "atol": 1e-12, "reserved_images_opened": 0}
        save_json(dest / "compatibility.json", compatibility)
        parents = [{"selection_path": str(Path(p).resolve()), "selection_sha256": file_hash(p),
                    "comparison_path": str(report), "comparison_sha256": file_hash(report)}
                   for p, report in ((stage4, old_report_path), (stage5, new_report_path))]
        bundle = {"kind": "bundle_v1", "selected": {**old["selected"], **new["selected"]},
                  "sources": sources, "outer_manifest_hash": outer["manifest_hash"],
                  "inner_signature": inner["signature"], "representations": REPRESENTATIONS,
                  "target_fpr": TARGET_FPR, "cache_hashes": new["cache_hashes"],
                  "comparison_path": str(new_report_path), "comparison_sha256": file_hash(new_report_path),
                  "parent_selections": parents, "compatibility_sha256": file_hash(dest / "compatibility.json"),
                  "final_calibration_executed": False}
        bundle["signature"] = signature(bundle)
        save_json(dest / "selection.json", bundle)
        print(f"Seleção dos quatro modelos preparada em {dest}; seis regressões verificadas; reservas não abertas.")
        return dest
    except Exception as exc:
        save_json(dest / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "cache", "stage4", "stage5", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    prepare_bundle(args.study, args.cache, args.stage4, args.stage5, args.out)


if __name__ == "__main__":
    main()
