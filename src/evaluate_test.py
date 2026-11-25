"""Avaliação única dos modelos finais calibrados nas 160 imagens de teste.

Não ajusta, não recalibra e não seleciona nada: só aplica representação,
detector e limiar já congelados por `src.final_calibration`. Requer opção
explícita, pois consome o teste.
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits

from .extract_features import processar_imagem
from .feature_experiments import file_hash, load_study, reserve_directory, save_json
from .features import HOG_CONFIGS, criar_banco_gabor
from .model_experiments import REPRESENTATIONS

MODELS = ("svm", "knn", "mahalanobis", "isolation_forest")


def metricas(y_true, y_pred, scores, classes):
    """y_true/y_pred em +1 (normal) / -1 (defeito); scores crescem com a anomalia."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # Defeito é a classe positiva: FN = defeito aprovado como bom.
    tn, fp, fn, tp = confusion_matrix(y_true == -1, y_pred == -1, labels=[False, True]).ravel()
    por_classe = {}
    for nome in sorted(set(classes) - {"good"}):
        mask = np.asarray(classes) == nome
        por_classe[nome] = {"n": int(mask.sum()), "detectados": int((y_pred[mask] == -1).sum()),
                            "recall": float((y_pred[mask] == -1).mean())}
    return {"tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
            "accuracy": float((tp + tn) / len(y_true)),
            "recall_defeitos": float(tp / (tp + fn)),
            "especificidade": float(tn / (tn + fp)),
            "falsos_negativos": int(fn), "taxa_falso_negativo": float(fn / (tp + fn)),
            "precisao_defeitos": float(tp / (tp + fp)) if tp + fp else None,
            "auc": float(roc_auc_score(y_true == -1, scores)),
            "por_categoria": por_classe}


def extrair_teste(paths, config):
    bank = criar_banco_gabor(orientations=4, lambdas=[8.0, 12.0, 16.0])
    vectors = []
    for path in paths:
        vector, ok = processar_imagem(Path(path), bank, hog_config=config)
        if not ok or vector is None or vector.shape != (config.total_dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"Características inválidas no teste: {path}")
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32)


def evaluate(study, final_root, out, *, confirm=False, suffix="v1"):
    if not confirm:
        raise ValueError("A avaliação consome o teste; use --confirm-test-evaluation.")
    if Path(out).exists() or Path(out).is_symlink():
        raise FileExistsError(f"Destino existente: {out}")
    inventory, outer, _ = load_study(study)
    info = inventory.set_index("caminho")
    ids = outer["test_ids"]
    for path in ids:
        if file_hash(path) != info.loc[path, "sha256"]:
            raise ValueError(f"Imagem de teste alterada: {path}")
    classes = [info.loc[p, "classe"] for p in ids]
    y_true = np.array([1 if c == "good" else -1 for c in classes])

    bundles = {}
    for model in MODELS:
        folder = Path(final_root) / f"final_{model}_{suffix}"
        meta = json.loads((folder / "final_calibration.json").read_text())
        if meta.get("test_accessed") or meta["outer_manifest_hash"] != outer["manifest_hash"]:
            raise ValueError(f"Calibração de {model} incompatível com o manifesto.")
        if file_hash(folder / "model.joblib") != meta["model_sha256"]:
            raise ValueError(f"model.joblib de {model} alterado.")
        bundles[model] = joblib.load(folder / "model.joblib")

    dest = reserve_directory(out)
    try:
        with threadpool_limits(limits=1):
            raw = {}
            for model, bundle in bundles.items():
                hog = REPRESENTATIONS[bundle["selection"]["representation"]]["hog"]
                if hog not in raw:
                    raw[hog] = extrair_teste(ids, HOG_CONFIGS[hog])
                    with (dest / f"test_features_{hog}.npz").open("xb") as stream:
                        np.savez_compressed(stream, features=raw[hog], ids=np.asarray(ids))
            resultados, linhas = {}, []
            for model, bundle in bundles.items():
                hog = REPRESENTATIONS[bundle["selection"]["representation"]]["hog"]
                X = bundle["representation"].transform(raw[hog])
                detector = bundle["detector"]
                scores, pred = detector.anomaly_score(X), detector.predict(X)
                resultados[model] = {"representacao": bundle["selection"]["representation"],
                                     "parametros": bundle["selection"]["parameters"],
                                     "limiar": detector.threshold_, **metricas(y_true, pred, scores, classes)}
                linhas += [{"modelo": model, "caminho": p, "classe": c, "predito": int(y),
                            "anomaly_score": float(s)} for p, c, y, s in zip(ids, classes, pred, scores)]
        save_json(dest / "test_evaluation.json", {"status": "complete", "n_test": len(ids),
                  "classes": {c: classes.count(c) for c in sorted(set(classes))}, "results": resultados})
        with (dest / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(linhas[0]))
            writer.writeheader()
            writer.writerows(linhas)
        with (dest / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["modelo", "representacao", "recall_defeitos", "falsos_negativos",
                             "especificidade", "falsos_positivos", "accuracy", "auc"])
            for model, r in sorted(resultados.items(), key=lambda kv: (kv[1]["fn"], -kv[1]["especificidade"])):
                writer.writerow([model, r["representacao"], f"{r['recall_defeitos']:.4f}", r["fn"],
                                 f"{r['especificidade']:.4f}", r["fp"], f"{r['accuracy']:.4f}", f"{r['auc']:.4f}"])
    except Exception as exc:
        save_json(dest / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--final-root", required=True, type=Path,
                        help="Pasta que contém final_<modelo>_<sufixo>/")
    parser.add_argument("--suffix", default="v1")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--confirm-test-evaluation", action="store_true")
    args = parser.parse_args(argv)
    try:
        dest = evaluate(args.study, args.final_root, args.out, confirm=args.confirm_test_evaluation,
                        suffix=args.suffix)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Avaliação interrompida: {exc}\n")
    print((dest / "summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
