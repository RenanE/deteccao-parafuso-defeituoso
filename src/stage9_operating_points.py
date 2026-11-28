"""Análise exploratória dos pontos de operação locais, com holdout já consultado.

Ajusta somente as cinco configurações congeladas da etapa 9. Execute na raiz:
    python -m src.stage9_operating_points --out results/stage9/operating_points_v1
"""

import argparse
import csv
import json
from os.path import commonpath
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import stage9_experiments as stage9
from .defect_split import make_split
from .feature_experiments import file_hash, save_json, signature
from .local_features import PatchAnomalyDetector, extrair_lote
from .validation import validar_divisao

STUDY = Path("results/stage3/study_v1")
GROUPS = (("selection_v1", "final_v1"), ("selection_knn_mahalanobis_v1", "final_knn_mahalanobis_v1"))
STRATEGIES = {
    "original": ("Quantil 95%", "#1b64be", "o"),
    "selection_95": ("Recall seleção ≥ 95%", "#138570", "s"),
    "selection_100": ("Recall seleção = 100%", "#a3512a", "^"),
}
LABELS = {"svm": "One-Class SVM", "lof": "LOF", "isolation_forest": "Isolation Forest",
          "knn": "kNN", "mahalanobis": "Mahalanobis"}


def recall_threshold(defect_scores, percent):
    """Maior float64 que detecta ao menos ceil(n * percent / 100) defeitos."""
    scores = np.asarray(defect_scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Use um vetor não vazio de scores finitos.")
    if not isinstance(percent, int) or isinstance(percent, bool) or not 1 <= percent <= 100:
        raise ValueError("Percentual deve ser um inteiro entre 1 e 100.")
    required = (scores.size * percent + 99) // 100
    cutoff = np.partition(scores, scores.size - required)[scores.size - required]
    return float(np.nextafter(cutoff, -np.inf))


def determine_thresholds(calibration, selection_defects):
    return {
        "original": float(np.quantile(calibration, 0.95, method="higher")),
        "selection_95": recall_threshold(selection_defects, 95),
        "selection_100": recall_threshold(selection_defects, 100),
    }


def evaluate(threshold, cal, sel, good, bad, categories, auc_holdout):
    tp, fp = int(np.sum(bad > threshold)), int(np.sum(good > threshold))
    by_category = {}
    for cat in sorted(set(categories)):
        detected = bad[categories == cat] > threshold
        by_category[cat] = {"n": len(detected), "detectados": int(detected.sum()),
                            "recall": float(detected.mean())}
    return {
        "threshold": threshold,
        "calibration_fpr": float(np.mean(cal > threshold)),
        "selection_tp": int(np.sum(sel > threshold)),
        "selection_recall": float(np.mean(sel > threshold)),
        "tp": tp, "tn": len(good) - fp, "fp": fp, "fn": len(bad) - tp,
        "recall_defeitos": tp / len(bad), "especificidade": (len(good) - fp) / len(good),
        "auc_holdout": auc_holdout, "por_categoria": by_category,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_original(model, selected, reference, old_predictions, recovered, expected_pred, original, extra_auc):
    if set(old_predictions) != set(recovered):
        raise ValueError(f"{model}: IDs das predições originais divergem.")
    old_scores = np.array([float(old_predictions[p]["score"]) for p in recovered])
    new_scores = np.array(list(recovered.values()))
    np.testing.assert_allclose(new_scores, old_scores, rtol=0, atol=1e-10, err_msg=model)
    if any(int(r["predito"]) != expected_pred[p] for p, r in old_predictions.items()):
        raise ValueError(f"{model}: predições originais divergem.")
    for key in ("config", "params"):
        if reference[key] != selected[key]:
            raise ValueError(f"{model}: {key} diverge da avaliação original.")
    for key in ("tp", "tn", "fp", "fn", "calibration_fpr", "recall_defeitos", "especificidade"):
        if original[key] != reference[key]:
            raise ValueError(f"{model}: {key} diverge da avaliação original.")
    for key in ("threshold", "auc_holdout"):
        np.testing.assert_allclose(original[key], reference[key], rtol=0, atol=1e-10, err_msg=f"{model}: {key}")
    np.testing.assert_allclose(extra_auc, reference["auc_reservadas_vs_defeitos_holdout"], rtol=0, atol=1e-10)
    counts = {cat: {k: v for k, v in m.items() if k != "recall"}
              for cat, m in original["por_categoria"].items()}
    if counts != reference["por_categoria"]:
        raise ValueError(f"{model}: métricas por categoria divergem.")
    return float(np.max(np.abs(new_scores - old_scores)))


def plot_results(out, results, roc_inputs):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e7e7e7"})
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for (model, thresholds), ax in zip(results.items(), axes.flat):
        good, bad = roc_inputs[model]
        fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(good)), np.ones(len(bad))], np.r_[good, bad])
        ax.plot(fpr, tpr, color="#45596b", label=f"AUC {thresholds['original']['auc_holdout']:.4f}")
        ax.plot([0, 1], [0, 1], color="#c2c2c2", linestyle="--")
        for strategy, (label, color, marker) in STRATEGIES.items():
            point = thresholds[strategy]
            ax.scatter(point["fp"] / len(good), point["recall_defeitos"], label=label,
                       edgecolor=color, marker=marker, facecolor="none", linewidth=1.7)
        ax.set(title=LABELS[model], xlabel="Taxa de falsos positivos", ylabel="Recall",
               xlim=(-.03, 1.03), ylim=(-.03, 1.04))
        ax.grid(alpha=.6)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    axes.flat[-1].axis("off")
    fig.savefig(out / "roc_operating_points.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), layout="constrained")
    x = np.arange(len(results))
    for key, ax, title in zip(("fn", "fp"), axes, ("Defeitos aprovados (FN)", "Peças boas rejeitadas (FP)")):
        for pos, (strategy, (label, color, _)) in enumerate(STRATEGIES.items()):
            bars = ax.bar(x + .24 * (pos - 1), [r[strategy][key] for r in results.values()],
                          width=.24, label=label, color=color)
            ax.bar_label(bars, padding=2)
        ax.set_xticks(x, [LABELS[model] for model in results])
        ax.set_title(title)
        ax.margins(y=.22)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "fn_fp_comparison.png", dpi=160)
    plt.close(fig)


def run(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    stage_dir = stage9.CACHE.parent
    originals = [stage9.CACHE, stage9.SPLIT, *(STUDY / name for name in
                 ("inventory.json", "outer_manifest.json", "internal_manifest.json"))]
    for selection_dir, final_dir in GROUPS:
        originals.extend((stage_dir / selection_dir / "selection.json",
                          stage_dir / final_dir / "holdout_evaluation.json",
                          stage_dir / final_dir / "predictions.csv"))
    inputs = {p.as_posix(): file_hash(p) for p in originals}
    outer = read_json(STUDY / "outer_manifest.json")
    inner = read_json(STUDY / "internal_manifest.json")
    inventory = read_json(STUDY / "inventory.json")
    validar_divisao(outer, pd.DataFrame(inventory))
    if inner["outer_manifest_hash"] != outer["manifest_hash"] or signature(
        {k: v for k, v in inner.items() if k != "signature"}
    ) != inner["signature"]:
        raise ValueError("Manifesto interno incompatível.")
    cache, ids, idx, split = stage9.load()
    with cache:
        rois, masks = cache["rois"], cache["masks"]
    if ids != outer["fit_ids"] + outer["calibration_ids"] + outer["test_ids"]:
        raise ValueError("IDs do cache divergem do manifesto.")
    expected_split = make_split(inventory, outer["test_ids"])
    if (split["outer_manifest_hash"] != outer["manifest_hash"]
            or any(split[k] != v for k, v in expected_split.items())):
        raise ValueError("Divisão do teste diverge do manifesto e da semente.")
    expected_sizes = {"fit": 256, "calibration": 64, "sel_defects": 58, "sel_goods": 20,
                      "hold_defects": 61, "hold_goods": 21}
    if any(len(idx[k]) != n for k, n in expected_sizes.items()):
        raise ValueError("Tamanho de partição divergente.")

    models = {}
    for selection_dir, final_dir in GROUPS:
        selection = read_json(stage_dir / selection_dir / "selection.json")
        if (signature({k: v for k, v in selection.items() if k != "signature"}) != selection["signature"]
                or selection["split_signature"] != split["signature"]
                or selection["roi_cache_sha256"] != inputs[stage9.CACHE.as_posix()]):
            raise ValueError(f"Seleção incompatível: {selection_dir}.")
        for model, selected in selection["selected"].items():
            if model in models:
                raise ValueError(f"Modelo duplicado: {model}.")
            models[model] = (selected, final_dir, selection["signature"])
    if set(models) != set(LABELS):
        raise ValueError("São necessárias as cinco configurações congeladas.")

    # A pontuação por alvo usa somente o ajuste e a dobra de referência.
    # Os limiares recebem exclusivamente calibração e defeitos de seleção.
    target_keys = ("calibration", "sel_defects", "sel_goods", "hold_goods", "hold_defects")
    common = Path(commonpath(ids))
    relative = {p: Path(p).relative_to(common).as_posix() for p in ids}
    score_records, metrics, verification, roc_inputs = [], {}, {}, {}
    for model, (selected, final_dir, selection_sig) in models.items():
        cfg = stage9.config_from(selected)
        features = extrair_lote(rois, masks, cfg)
        detector = PatchAnomalyDetector(model, json.loads(selected["params"]), cfg, seed=42)
        maps = detector.fit_score(features[idx["fit"]], [features[idx[key]] for key in target_keys])
        del features
        scores = dict(zip(target_keys, map(detector.aggregate, maps)))
        if any(not np.isfinite(values).all() for values in scores.values()):
            raise ValueError(f"Scores não finitos: {model}.")
        cal, sel = scores["calibration"], scores["sel_defects"]
        thresholds = determine_thresholds(cal, sel)
        selection_auc = stage9.auc(cal, sel)
        selection_recall = stage9.recall_at_spec(cal, sel)
        np.testing.assert_allclose(selection_auc, selected["auc_sel"], rtol=0, atol=1e-12, err_msg=model)
        np.testing.assert_allclose(selection_recall, selected["recall95_sel"], rtol=0, atol=1e-12, err_msg=model)

        good, bad = scores["hold_goods"], scores["hold_defects"]
        roc_inputs[model] = good, bad
        categories = np.array([split["classes"][ids[i]] for i in idx["hold_defects"]])
        auc_holdout = stage9.auc(good, bad)
        points = {key: evaluate(thr, cal, sel, good, bad, categories, auc_holdout)
                  for key, thr in thresholds.items()}
        metrics[model] = points

        reference_path = stage_dir / final_dir
        original_report = read_json(reference_path / "holdout_evaluation.json")
        if original_report["selection_signature"] != selection_sig:
            raise ValueError(f"{model}: avaliação original vinculada a outra seleção.")
        with (reference_path / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            old_predictions = {r["caminho"]: r for r in csv.DictReader(stream) if r["modelo"] == model}
        hold_idx = np.r_[idx["hold_goods"], idx["hold_defects"]]
        recovered = {ids[i]: score for i, score in zip(hold_idx, np.r_[good, bad])}
        predicted = {p: (-1 if score > thresholds["original"] else 1) for p, score in recovered.items()}
        extra_auc = stage9.auc(cal, bad)
        verification[model] = {
            "selection_auc": selection_auc, "selection_recall_original": selection_recall,
            "auc_reservadas_vs_defeitos_holdout": extra_auc,
            "holdout_score_max_abs_error": check_original(
                model, selected, original_report["results"][model], old_predictions, recovered,
                predicted, points["original"], extra_auc,
            ),
            "threshold_abs_error": abs(thresholds["original"] - original_report["results"][model]["threshold"]),
        }
        for key, values in scores.items():
            for i, score in zip(idx[key], values):
                p = ids[i]
                score_records.append({"modelo": model, "particao": key, "id": relative[p],
                                      "classe": split["classes"].get(p, "good"), "score": float(score)})
        print(f"{model}: protocolo original verificado", flush=True)

    if any(file_hash(p) != inputs[p.as_posix()] for p in originals):
        raise ValueError("Artefatos originais alterados durante a análise.")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "scores.csv", score_records)
    write_csv(out / "metrics.csv", [
        {"modelo": model, "estrategia": strategy, **{k: v for k, v in point.items() if k != "por_categoria"}}
        for model, points in metrics.items() for strategy, point in points.items()
    ])
    write_csv(out / "category_recall.csv", [
        {"modelo": model, "estrategia": strategy, "categoria": cat, **values}
        for model, points in metrics.items() for strategy, point in points.items()
        for cat, values in point["por_categoria"].items()
    ])
    save_json(out / "analysis.json", {
        "exploratoria": True, "fit_seed": 42, "particoes": expected_sizes, "inputs_sha256": inputs,
        "verificacao_original": verification, "pontos": metrics,
    })
    plot_results(out, metrics, roc_inputs)
    (out / "README.md").write_text(
        "# Pontos de operação da etapa 9\n\n"
        "Análise complementar e exploratória. O holdout já foi consultado nas avaliações anteriores. "
        "As cinco configurações foram reajustadas sobre as mesmas 256 imagens boas com ajuste cruzado. "
        "Os limiares foram definidos apenas com a calibração ou os defeitos de seleção. "
        "As 20 boas da metade de seleção aparecem em scores.csv para registro.\n\n"
        "A decisão usa score > limiar. Para os alvos de seleção, usa-se o maior float64 abaixo "
        "do score de corte, incluindo empates. O limiar original é o quantil 95% superior das boas reservadas. "
        "A AUC usa os scores do holdout e é constante entre limiares de cada modelo.\n\n"
        "Os scores de calibração e seleção foram persistidos aqui pela primeira vez. A verificação "
        "compara o limiar original, as métricas de seleção, os scores e predições do holdout e as métricas finais.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)


if __name__ == "__main__":
    main()
