"""Otimização exploratória de limiares a partir dos scores salvos da etapa 9."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .stage9_operating_points import LABELS, determine_thresholds, write_csv
from .stage9_experiments import auc

WEIGHTS = (1, 2, 3, 5, 10)
PARTITION_SIZES = {"calibration": 64, "sel_defects": 58, "sel_goods": 20,
                   "hold_goods": 21, "hold_defects": 61}


def fit_thresholds(calibration, selection):
    """Minimiza o custo exato; desempata por menor FNR e maior limiar observado."""
    good, bad = (np.sort(np.asarray(v, dtype=np.float64)) for v in (calibration, selection))
    if any(v.ndim != 1 or not v.size or not np.isfinite(v).all() for v in (good, bad)):
        raise ValueError("Calibração e seleção devem ser vetores não vazios de scores finitos.")
    values = np.union1d(good, bad)
    candidates = np.r_[np.nextafter(values[0], -np.inf), values]
    # A busca à direita mantém todo empate com o limiar na classe normal.
    fn = np.searchsorted(bad, candidates, side="right")
    fp = len(good) - np.searchsorted(good, candidates, side="right")
    points = {}
    for w in WEIGHTS:
        # Multiplicar pelo denominador comum preserva empates exatamente.
        numerator = w * fn * len(good) + fp * len(bad)
        best = int(np.lexsort((-candidates, fn, numerator))[0])
        points[w] = float(candidates[best])
    return points


def nondominated(rows):
    pairs = {(r["fn"], r["fp"]) for r in rows}
    return {(fn, fp) for fn, fp in pairs if not any(
        (other_fn, other_fp) != (fn, fp) and other_fn <= fn and other_fp <= fp
        for other_fn, other_fp in pairs
    )}


def read_scores(source):
    raw = Path(source).read_bytes()
    by_model = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")):
        group = by_model.setdefault(row["modelo"], {}).setdefault(row["particao"], {})
        if row["id"] in group:
            raise ValueError("ID repetido na partição.")
        expected_normal = row["particao"] in ("calibration", "sel_goods", "hold_goods")
        value = float(row["score"])
        if (row["classe"] == "good") != expected_normal or not np.isfinite(value):
            raise ValueError("Classe incompatível com a partição ou score inválido.")
        group[row["id"]] = value
    if set(by_model) != set(LABELS):
        raise ValueError("O arquivo deve conter scores dos cinco modelos locais.")
    reference_ids = None
    scores = {}
    for model, groups in by_model.items():
        if {name: len(rows) for name, rows in groups.items()} != PARTITION_SIZES:
            raise ValueError(f"Partições incompletas: {model}.")
        membership = {name: set(rows) for name, rows in groups.items()}
        if len(set().union(*membership.values())) != sum(PARTITION_SIZES.values()):
            raise ValueError(f"Partições sobrepostas: {model}.")
        if reference_ids is not None and reference_ids != membership:
            raise ValueError(f"IDs divergentes entre modelos: {model}.")
        reference_ids = membership
        scores[model] = {name: np.array(list(rows.values())) for name, rows in groups.items()}
    return scores, hashlib.sha256(raw).hexdigest()


def compare_points(scores):
    # Todos os limiares são congelados antes de avaliar qualquer métrica de holdout.
    thresholds = {}
    for model, groups in scores.items():
        cal, sel = groups["calibration"], groups["sel_defects"]
        thresholds[model] = {
            **determine_thresholds(cal, sel),
            **{f"cost_w{w}": threshold for w, threshold in fit_thresholds(cal, sel).items()},
        }

    rows = []
    for model, groups in scores.items():
        cal, sel = groups["calibration"], groups["sel_defects"]
        good, bad = groups["hold_goods"], groups["hold_defects"]
        roc_auc = auc(good, bad)
        for strategy, threshold in thresholds[model].items():
            sel_fnr = np.mean(sel <= threshold)
            cal_fpr = np.mean(cal > threshold)
            fn, fp = int(np.sum(bad <= threshold)), int(np.sum(good > threshold))
            fnr, fpr = fn / len(bad), fp / len(good)
            rows.append({
                "modelo": model, "estrategia": strategy, "limiar": threshold,
                "fnr_selecao": sel_fnr, "fpr_calibracao": cal_fpr,
                "fn": fn, "fp": fp, "recall": (len(bad) - fn) / len(bad),
                "especificidade": (len(good) - fp) / len(good), "auc": roc_auc,
                **{f"j_selecao_w{w}": w * sel_fnr + cal_fpr for w in WEIGHTS},
                **{f"j_holdout_w{w}": w * fnr + fpr for w in WEIGHTS},
            })
    return rows


def plot_pareto(out, rows, global_front):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for model, label in LABELS.items():
        points = {(r["fn"], r["fp"]) for r in rows if r["modelo"] == model}
        ax.scatter([fn for fn, _ in points], [fp for _, fp in points], alpha=.75, label=label)
    fn, fp = zip(*sorted(global_front))
    ax.step(fn, fp, where="post", color="#263744", linestyle="--", label="Fronteira observada")
    ax.scatter(fn, fp, edgecolors="#263744", facecolors="none", s=130, linewidths=1.4)
    for x, y in zip(fn, fp):
        ax.annotate(f"({x}, {y})", (x, y), xytext=(5, 6), textcoords="offset points")
    ax.set(xlabel="Falsos negativos (FN)", ylabel="Falsos positivos (FP)",
           title="Fronteira de Pareto dos pontos avaliados no holdout")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(out / "pareto_holdout.png", dpi=160)
    plt.close(fig)


def run(source, out):
    if out.exists():
        raise FileExistsError(out)
    scores, source_hash = read_scores(source)
    rows = compare_points(scores)
    global_front = nondominated(rows)
    model_fronts = {model: nondominated([r for r in rows if r["modelo"] == model]) for model in scores}
    pareto = [
        {"modelo": r["modelo"], "estrategia": r["estrategia"], "fn": r["fn"], "fp": r["fp"],
         "pareto_modelo": (r["fn"], r["fp"]) in model_fronts[r["modelo"]],
         "pareto_global": (r["fn"], r["fp"]) in global_front}
        for r in rows if (r["fn"], r["fp"]) in model_fronts[r["modelo"]]
    ]
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "comparison.csv", rows)
    write_csv(out / "pareto.csv", pareto)
    plot_pareto(out, rows, global_front)
    (out / "provenance.json").write_text(json.dumps({
        "scores_sha256": source_hash, "weights": WEIGHTS, "partition_sizes": PARTITION_SIZES,
        "exploratory": True,
    }, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Otimização exploratória de limiares\n\n"
        "O holdout já havia sido consultado. Os pesos foram fixados em 1, 2, 3, 5 e 10. "
        "O ajuste de J = w × FNR + FPR usou apenas 58 defeitos de seleção e 64 boas de calibração. "
        "Nenhuma escolha de peso ou limiar usa o holdout.\n\n"
        "A decisão é score > limiar. Os candidatos são os scores distintos do ajuste de limiar "
        "e o float64 anterior ao mínimo, que permite rejeitar todos. Em empate de custo, "
        "vence a menor FNR e, se persistir, o maior limiar candidato. "
        "O máximo observado representa a decisão de aprovar todos no ajuste de limiar.\n\n"
        "comparison.csv reporta as taxas e os custos nos cinco pesos para os três pontos anteriores "
        "e para os cinco novos por modelo. AUC é calculada dos scores do holdout e independe do limiar. "
        "pareto.csv inclui pontos não dominados por modelo e indica os também não dominados globalmente "
        "entre os 40 pontos avaliados. Empates de FN e FP preservam todas as estratégias equivalentes.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.scores, args.out)


if __name__ == "__main__":
    main()
