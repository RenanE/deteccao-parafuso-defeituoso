"""Procedimento posterior: ajuste nas boas de desenvolvimento e calibração final.

Não seleciona parâmetros e não lê imagens de teste. Requer opção explícita.
Este comando NÃO faz parte da execução da busca da etapa 4.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .extract_features import processar_imagem
from .feature_experiments import file_hash, reserve_directory, save_json, signature
from .features import HOG_CONFIGS, criar_banco_gabor
from .model_experiments import (
    REPRESENTATIONS, TARGET_FPR, load_development, make_detector,
    make_representation, protocol_sources, resources,
)


def finalize(study, cache, selection_path, out, *, model_name, confirm=False):
    if not confirm:
        raise ValueError("Calibração final exige --confirm-final-calibration; não executar durante a seleção.")
    if model_name not in ("svm", "knn", "mahalanobis", "isolation_forest"):
        raise ValueError("Modelo desconhecido.")
    if Path(out).exists() or Path(out).is_symlink():
        raise FileExistsError(f"Destino existente: {out}")
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text())
    unsigned = {k: v for k, v in selection.items() if k != "signature"}
    if signature(unsigned) != selection["signature"]:
        raise ValueError("Seleção alterada; assinatura inválida.")
    if selection["sources"] != protocol_sources() or selection["representations"] != REPRESENTATIONS:
        raise ValueError("Código/protocolo mudou desde a seleção; revisar antes da calibração final.")
    comparison_path = Path(selection.get("comparison_path", selection_path.parent / "comparison.json"))
    if file_hash(comparison_path) != selection["comparison_sha256"]:
        raise ValueError("Relatório da seleção alterado.")
    if selection.get("kind") == "bundle_v1":
        if file_hash(selection_path.parent / "compatibility.json") != selection["compatibility_sha256"]:
            raise ValueError("Registro de compatibilidade alterado.")
        for parent in selection["parent_selections"]:
            if (file_hash(parent["selection_path"]) != parent["selection_sha256"]
                    or file_hash(parent["comparison_path"]) != parent["comparison_sha256"]):
                raise ValueError("Seleção/relatório histórico de origem alterado.")
    if selection["target_fpr"] != TARGET_FPR:
        raise ValueError("FPR-alvo difere do protocolo selecionado.")
    inventory, outer, inner, matrices, metadata = load_development(study, cache)
    if (selection["outer_manifest_hash"] != outer["manifest_hash"]
            or selection["inner_signature"] != inner["signature"]
            or selection["cache_hashes"] != {k: v["cache_sha256"] for k, v in metadata.items()}):
        raise ValueError("Seleção incompatível com o estudo/cache.")
    limits = resources()
    if (limits["disk_free_bytes"] < limits["estimated_disk_budget_bytes"]
            or (limits["available_memory_bytes"] is not None
                and limits["available_memory_bytes"] < limits["estimated_working_budget_bytes"])):
        raise OSError("Recursos insuficientes para a calibração final.")
    choice = selection["selected"][model_name]
    name = choice["representation"]
    spec = REPRESENTATIONS[name]
    config = HOG_CONFIGS[spec["hog"]]
    hashes = inventory.set_index("caminho")["sha256"]
    ids = outer["calibration_ids"]
    # Somente aqui, após a opção explícita, a reserva final é aberta.
    for path in ids:
        if file_hash(path) != hashes[path]:
            raise ValueError(f"Imagem da calibração final alterada: {path}")
    dest = reserve_directory(out)
    try:
        with threadpool_limits(limits=1):
            bank = criar_banco_gabor(orientations=4, lambdas=[8.0, 12.0, 16.0])
            vectors = []
            for path in ids:
                vector, ok = processar_imagem(Path(path), bank, hog_config=config)
                if (not ok or vector is None or vector.shape != (config.total_dimension,)
                        or not np.isfinite(vector).all() or file_hash(path) != hashes[path]):
                    raise ValueError(f"Características/imagem inválidas na calibração: {path}")
                vectors.append(vector)
            raw = np.asarray(vectors, dtype=np.float32)
            representation = make_representation(name, seed=42)
            development = representation.fit_transform(matrices[spec["hog"]])
            calibration = representation.transform(raw)
            detector = make_detector(choice).fit(development, sample_ids=outer["fit_ids"])
            detector.calibrate(calibration, sample_ids=ids)
            bundle = {"representation": representation, "detector": detector,
                      "selection": choice, "outer_manifest_hash": outer["manifest_hash"]}
            with (dest / "model.joblib").open("xb") as stream:
                joblib.dump(bundle, stream)
            with (dest / "calibration_features.npz").open("xb") as stream:
                np.savez_compressed(stream, features=raw, ids=np.asarray(ids),
                                    image_hashes=np.asarray([hashes[p] for p in ids]))
            save_json(dest / "final_calibration.json", {
                "status": "complete", "model": model_name, "selection": choice,
                "selection_sha256": file_hash(selection_path), "sources": protocol_sources(),
                "outer_manifest_hash": outer["manifest_hash"], "n_fit": len(development),
                "n_calibration": len(calibration), "fit_ids": outer["fit_ids"], "calibration_ids": ids,
                "descriptor_config": config.metadata(), "seed": 42, "target_fpr": TARGET_FPR,
                "threshold": detector.threshold_, "calibration_fpr": detector.calibration_fpr_,
                "calibration_scores": detector.calibration_scores_.tolist(),
                "model_sha256": file_hash(dest / "model.joblib"),
                "calibration_cache_sha256": file_hash(dest / "calibration_features.npz"),
                "test_accessed": False})
    except Exception as exc:
        save_json(dest / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("svm", "knn", "mahalanobis", "isolation_forest"))
    parser.add_argument("--confirm-final-calibration", action="store_true")
    args = parser.parse_args(argv)
    try:
        finalize(args.study, args.cache, args.selection, args.out,
                 model_name=args.model, confirm=args.confirm_final_calibration)
    except (ValueError, OSError, MemoryError) as exc:
        parser.exit(1, f"Calibração final interrompida: {exc}\n")


if __name__ == "__main__":
    main()
