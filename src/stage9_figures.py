"""Figuras da etapa 9 (comparação global x local, ablação, mapas locais).

    python -m src.stage9_figures --out results/stage9/figures
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from .final_evaluation import FINAL_MODELS
from .preprocessing import aplicar_clahe

MODELS = ("svm", "lof", "isolation_forest")
LABEL = {"svm": "One-Class SVM", "lof": "LOF", "isolation_forest": "Isolation Forest"}
BEFORE, AFTER = "#9dc8f7", "#1b64be"
STYLE = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c2c2c2",
         "xtick.color": "#666666", "ytick.color": "#666666", "grid.color": "#e7e7e7"}


def baseline(split):
    """Modelos globais congelados (etapas 5/6), re-pontuados na etapa 8, nas mesmas metades."""
    rows = list(csv.DictReader(open("results/stage8/test_evaluation_classical_v1/predictions.csv", encoding="utf-8")))
    sel, hold, cls = set(split["selection_ids"]), set(split["holdout_ids"]), split["classes"]
    out = {}
    for m in MODELS:
        meta = json.loads((Path(FINAL_MODELS[m][0]) / "final_calibration.json").read_text())
        cal, thr = np.array(meta["calibration_scores"]), meta["threshold"]
        s = {r["caminho"]: float(r["anomaly_score"]) for r in rows if r["modelo"] == m}
        sd = np.array([s[p] for p in sel if cls[p] != "good"])
        hd = np.array([s[p] for p in hold if cls[p] != "good"])
        hg = np.array([s[p] for p in hold if cls[p] == "good"])
        y = lambda n, p: roc_auc_score(np.r_[np.zeros(len(n)), np.ones(len(p))], np.r_[n, p])
        out[m] = {"auc_sel": y(cal, sd), "fn": int((hd <= thr).sum()), "fp": int((hg > thr).sum()),
                  "tp": int((hd > thr).sum()), "tn": int((hg <= thr).sum()),
                  "auc_holdout": y(hg, hd), "auc_reservadas_vs_defeitos_holdout": y(cal, hd),
                  "n_def": len(hd), "n_good": len(hg)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", default="results/stage9/selection_v1")
    parser.add_argument("--final-dir", default="results/stage9/final_v1")
    parser.add_argument("--out", default="results/stage9/figures")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(STYLE)
    split = json.loads(Path("results/stage9/defect_split_v1.json").read_text())
    base = baseline(split)
    final = json.loads((Path(args.final_dir) / "holdout_evaluation.json").read_text())["results"]
    selection = json.loads((Path(args.selection_dir) / "selection.json").read_text())["selected"]
    runs = list(csv.DictReader(open(Path(args.selection_dir) / "runs.csv")))
    json.dump({"baseline_global": base, "local": final}, open(out / "comparison.json", "w"), indent=1, ensure_ascii=False)

    # 1. Antes x depois: AUC (seleção), FN e FP no holdout
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    y = np.arange(len(MODELS))
    panels = [("AUC (reservadas vs defeitos de seleção)", [base[m]["auc_sel"] for m in MODELS],
               [selection[m]["auc_sel"] for m in MODELS], (0, 1), "{:.2f}"),
              (f"Falsos negativos no holdout (de {base['svm']['n_def']})", [base[m]["fn"] for m in MODELS],
               [final[m]["fn"] for m in MODELS], None, "{:d}"),
              (f"Falsos positivos no holdout (de {base['svm']['n_good']})", [base[m]["fp"] for m in MODELS],
               [final[m]["fp"] for m in MODELS], None, "{:d}")]
    for ax, (title, b, a, lim, fmt) in zip(axes, panels):
        ax.barh(y - .2, b, .38, color=BEFORE, label="Global (etapas 5/6)")
        ax.barh(y + .2, a, .38, color=AFTER, label="Local (etapa 9)")
        top = lim[1] if lim else max(max(b), max(a)) * 1.25 + 1
        for yy, v in zip(y - .2, b):
            ax.text(v + top * .01, yy, fmt.format(v), va="center", fontsize=8.5)
        for yy, v in zip(y + .2, a):
            ax.text(v + top * .01, yy, fmt.format(v), va="center", fontsize=8.5)
        ax.set_yticks(y, [LABEL[m] for m in MODELS])
        ax.invert_yaxis()
        ax.set_xlim(0, top * 1.08)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", lw=.5)
    axes[0].axvline(.5, color="#c2c2c2", ls="--", lw=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "before_after.png", dpi=170, facecolor="white")
    plt.close(fig)

    # 2. Ablação dos descritores (LOF k=20, patch 32)
    abl = [r for r in runs if r["tag"].startswith("ablacao")]
    names = []
    for r in abl:
        if r["tag"] == "ablacao_sem_znorm":
            names.append("todos, sem normalizar brilho")
        elif r["tag"] == "ablacao_sem_vizinhos":
            names.append("todos, sem tolerância a vizinhos")
        else:
            pt = {"hog": "HOG", "lbp": "LBP", "gabor": "Gabor", "edges": "bordas", "shape": "silhueta"}
            parts = r["descriptors"].split("+")
            names.append("todos" if len(parts) == 5 else " + ".join(pt[x] for x in parts))
    vals = [float(r["auc_sel"]) for r in abl]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    yy = np.arange(len(abl))
    ax.barh(yy, vals, color=AFTER, height=.6)
    for i, v in enumerate(vals):
        ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(yy, names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(.4, 1.03)
    ax.axvline(.5, color="#c2c2c2", ls="--", lw=1)
    ax.set_xlabel("AUC (reservadas vs defeitos de seleção)")
    ax.set_title("Ablação: LOF local, patch 32, k = 20", fontsize=10)
    ax.grid(True, axis="x", lw=.5)
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=170, facecolor="white")
    plt.close(fig)

    # 3. Curvas ROC no holdout: global (tracejado) x local (contínuo)
    from sklearn.metrics import roc_curve
    colors = {"svm": "#257de0", "lof": "#f47230", "isolation_forest": "#00b894"}
    hold = set(split["holdout_ids"])
    glob_rows = list(csv.DictReader(open("results/stage8/test_evaluation_classical_v1/predictions.csv", encoding="utf-8")))
    loc_rows = list(csv.DictReader(open(Path(args.final_dir) / "predictions.csv", encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for m in MODELS:
        for rows_, key, ls, tag in ((glob_rows, "anomaly_score", "--", "global"), (loc_rows, "score", "-", "local")):
            rr = [r for r in rows_ if r["modelo"] == m and r["caminho"] in hold]
            yv = [r["classe"] != "good" for r in rr]
            fpr, tpr, _ = roc_curve(yv, [float(r[key]) for r in rr])
            a = roc_auc_score(yv, [float(r[key]) for r in rr])
            ax.plot(fpr, tpr, ls=ls, lw=2 if tag == "local" else 1.4, color=colors[m],
                    label=f"{LABEL[m]} {tag} (AUC {a:.3f})")
    ax.plot([0, 1], [0, 1], color="#c2c2c2", lw=1, ls=":")
    ax.set(xlabel="Taxa de falsos positivos (boas rejeitadas)", ylabel="Recall de defeitos", xlim=(0, 1), ylim=(0, 1.02),
           title="Curvas ROC no holdout (61 defeitos, 21 boas)")
    ax.grid(True, lw=.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_holdout.png", dpi=170, facecolor="white")
    plt.close(fig)

    # 4. Recall por categoria no holdout (modelos locais)
    cats = sorted(final["lof"]["por_categoria"])
    mat = np.array([[final[m]["por_categoria"][c]["detectados"] / final[m]["por_categoria"][c]["n"] for c in cats] for m in MODELS])
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i, m in enumerate(MODELS):
        for j, c in enumerate(cats):
            d = final[m]["por_categoria"][c]
            ax.text(j, i, f"{d['detectados']}/{d['n']}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if mat[i, j] > .55 else "#191919")
    ax.set_xticks(range(len(cats)), cats, rotation=15, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(MODELS)), [LABEL[m] + " local" for m in MODELS], fontsize=9)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label="Recall")
    ax.set_title("Defeitos detectados por categoria no holdout", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "recall_by_category_holdout.png", dpi=170, facecolor="white")
    plt.close(fig)

    # 5. Mapas de anomalia locais sobre a ROI alinhada (modelo LOF)
    cache = np.load("results/stage9/roi_cache_v1.npz")
    pos = {str(p): i for i, p in enumerate(cache["ids"])}
    ex = np.load(Path(args.final_dir) / "example_maps.npz")
    m = "lof"
    ids, maps, thr, P = [str(x) for x in ex[f"{m}_ids"]], ex[f"{m}_maps"], float(ex[f"{m}_threshold"]), int(ex[f"{m}_patch"])
    fig, axes = plt.subplots(len(ids), 1, figsize=(7.2, 1.45 * len(ids)))
    vmax = max(4.0, float(maps.max()))
    for ax, p, amap in zip(axes, ids, maps):
        i = pos[p]
        roi = aplicar_clahe(cache["rois"][i], clip_limit=2.0, tile_grid_size=(8, 8))
        S = P // 2
        heat = np.zeros(roi.shape, np.float32)
        count = np.zeros(roi.shape, np.float32)
        for r in range(amap.shape[0]):
            for c in range(amap.shape[1]):
                heat[r * S:r * S + P, c * S:c * S + P] += amap[r, c]
                count[r * S:r * S + P, c * S:c * S + P] += 1
        heat = heat / np.maximum(count, 1)
        ax.imshow(roi, cmap="gray")
        ax.imshow(np.clip(heat, 0, None), cmap="inferno", alpha=.5, vmin=0, vmax=vmax)
        score = float(amap.max()) if selection[m]["aggregation"] == "max" else float(np.sort(amap.ravel())[-3:].mean())
        verdict = "DEFEITO" if score > thr else "bom"
        ax.set_title(f"{split['classes'][p]}: score {score:.1f} (limiar {thr:.1f}) -> {verdict}", fontsize=8.5, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "local_maps_lof.png", dpi=170, facecolor="white")
    plt.close(fig)

    # 6. Mesmos mapas em grade 3 x 2 (formato para o pôster)
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 4.4))
    for ax, p, amap in zip(axes.ravel(), ids, maps):
        i = pos[p]
        roi = aplicar_clahe(cache["rois"][i], clip_limit=2.0, tile_grid_size=(8, 8))
        S = P // 2
        heat = np.zeros(roi.shape, np.float32)
        count = np.zeros(roi.shape, np.float32)
        for r in range(amap.shape[0]):
            for c in range(amap.shape[1]):
                heat[r * S:r * S + P, c * S:c * S + P] += amap[r, c]
                count[r * S:r * S + P, c * S:c * S + P] += 1
        heat = heat / np.maximum(count, 1)
        ax.imshow(roi, cmap="gray")
        ax.imshow(np.clip(heat, 0, None), cmap="inferno", alpha=.5, vmin=0, vmax=vmax)
        score = float(np.sort(amap.ravel())[-3:].mean()) if selection[m]["aggregation"] == "top3" else float(amap.max())
        ax.set_title(f"{split['classes'][p]}: {'DEFEITO' if score > thr else 'bom'} ({score:.1f})", fontsize=10, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "local_maps_lof_grid.png", dpi=170, facecolor="white")
    plt.close(fig)
    print(json.dumps({m: {"antes": {k: base[m][k] for k in ("fn", "fp", "auc_holdout", "auc_reservadas_vs_defeitos_holdout")},
                          "depois": {k: final[m][k] for k in ("fn", "fp", "auc_holdout", "auc_reservadas_vs_defeitos_holdout")}}
                      for m in MODELS}, indent=1))


if __name__ == "__main__":
    main()
