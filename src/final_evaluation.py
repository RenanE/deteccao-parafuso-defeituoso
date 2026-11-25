"""Etapa 8: avaliação dos modelos clássicos finais (SVM, LOF, Isolation Forest) no teste.

Nada é ajustado, recalibrado ou selecionado aqui: aplica representação,
detector e limiar congelados na calibração final (etapas 5, 6 e 7).
Defeito é a classe positiva; FN = parafuso defeituoso aprovado como bom.

Uso:
    python -m src.final_evaluation --study results/stage3/study_v1 \
        --out results/stage8/test_evaluation_v1 --confirm-test-evaluation
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .evaluate_test import extrair_teste, metricas
from .feature_experiments import file_hash, load_study, reserve_directory, save_json
from .features import HOG_CONFIGS
from .model_experiments import REPRESENTATIONS

FINAL_MODELS = {
    # nome: (pasta, família, rótulo legível). Escopo final: SVM, LOF e Isolation Forest.
    # k-NN e Mahalanobis (etapas 4/5) e as redes neurais (neural_networks/) ficaram fora.
    "svm": ("results/stage5/final_svm_v1", "clássico", "One-Class SVM"),
    "lof": ("results/stage6/final_lof_v1", "clássico", "LOF"),
    "isolation_forest": ("results/stage5/final_isolation_forest_v1", "clássico", "Isolation Forest"),
}
CATEGORIES = ("manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top")
COLORS = ["#257de0", "#f47230", "#00b894"]


def post_hoc(y_true, scores):
    """Análise descritiva NO TESTE (não usada para escolher limiar ou modelo)."""
    defect = np.asarray(y_true) == -1
    s_def, s_good = scores[defect], scores[~defect]
    # Limiar mínimo que aprova zero defeitos: tudo >= min(score defeito) é rejeitado.
    fp_zero_fn = int(np.sum(s_good >= s_def.min()))
    # Recall com especificidade >= 95% (no máximo 2 das 41 boas rejeitadas).
    allowed = int(np.floor(0.05 * len(s_good)))
    thr = np.sort(s_good)[::-1][allowed] if allowed < len(s_good) else -np.inf
    return {"fp_para_zero_fn": fp_zero_fn, "fpr_para_zero_fn": fp_zero_fn / len(s_good),
            "recall_com_especificidade_95": float(np.mean(s_def > thr))}


def load_bundle(name, folder, outer):
    folder = Path(folder)
    meta = json.loads((folder / "final_calibration.json").read_text())
    if meta.get("test_accessed") or meta["outer_manifest_hash"] != outer["manifest_hash"]:
        raise ValueError(f"Calibração de {name} incompatível com o manifesto.")
    artifact = folder / "model.joblib"
    if file_hash(artifact) != meta["model_sha256"]:
        raise ValueError(f"Modelo de {name} alterado após a calibração.")
    bundle = joblib.load(artifact)
    return {**bundle, "meta": meta}


def evaluate(study, out, *, confirm=False):
    if not confirm:
        raise ValueError("A avaliação consome o teste; use --confirm-test-evaluation.")
    if Path(out).exists():
        raise FileExistsError(f"Destino existente: {out}")
    inventory, outer, _ = load_study(study)
    info = inventory.set_index("caminho")
    ids = outer["test_ids"]
    for path in ids:
        if file_hash(path) != info.loc[path, "sha256"]:
            raise ValueError(f"Imagem de teste alterada: {path}")
    classes = [info.loc[p, "classe"] for p in ids]
    y_true = np.array([1 if c == "good" else -1 for c in classes])
    bundles = {name: load_bundle(name, spec[0], outer) for name, spec in FINAL_MODELS.items()}

    dest = reserve_directory(out)
    results, rows, raw = {}, [], {}
    for name, bundle in bundles.items():
        print(f"Avaliando {name}...", flush=True)
        detector = bundle["detector"]
        representation = bundle["selection"]["representation"]
        parameters = bundle["selection"]["parameters"]
        hog = REPRESENTATIONS[representation]["hog"]
        if hog not in raw:
            with threadpool_limits(limits=1):
                raw[hog] = extrair_teste(ids, HOG_CONFIGS[hog])
            with (dest / f"test_features_{hog}.npz").open("xb") as stream:
                np.savez_compressed(stream, features=raw[hog], ids=np.asarray(ids))
        scores = detector.anomaly_score(bundle["representation"].transform(raw[hog]))
        pred = np.where(scores > detector.threshold_, -1, 1)  # empates = normal, como em predict
        results[name] = {"rotulo": FINAL_MODELS[name][2], "familia": FINAL_MODELS[name][1],
                         "representacao": representation, "parametros": parameters,
                         "limiar": float(detector.threshold_),
                         **metricas(y_true, pred, scores, classes),
                         "pos_hoc_no_teste": post_hoc(y_true, scores)}
        rows += [{"modelo": name, "caminho": p, "classe": c, "predito": int(y), "anomaly_score": float(s),
                  "score_sobre_limiar": float(s / detector.threshold_) if detector.threshold_ > 0 else None}
                 for p, c, y, s in zip(ids, classes, pred, scores)]

    ranking = sorted(results, key=lambda m: (results[m]["fn"], -results[m]["especificidade"], -results[m]["auc"]))
    save_json(dest / "test_evaluation.json", {
        "status": "complete", "n_test": len(ids),
        "classes": {c: classes.count(c) for c in sorted(set(classes))},
        "criterio": "Menor número de falsos negativos; desempate: maior especificidade, depois maior AUC.",
        "ranking": ranking, "results": results})
    with (dest / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (dest / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["posicao", "modelo", "familia", "representacao", "tp", "fn", "tn", "fp",
                         "recall_defeitos", "especificidade", "precisao", "acuracia", "auc",
                         *[f"recall_{c}" for c in CATEGORIES],
                         "pos_hoc_fp_para_zero_fn", "pos_hoc_recall_espec95"])
        for i, m in enumerate(ranking, 1):
            r = results[m]
            writer.writerow([i, m, r["familia"], r["representacao"], r["tp"], r["fn"], r["tn"], r["fp"],
                             f"{r['recall_defeitos']:.4f}", f"{r['especificidade']:.4f}",
                             f"{r['precisao_defeitos']:.4f}", f"{r['accuracy']:.4f}", f"{r['auc']:.4f}",
                             *[f"{r['por_categoria'][c]['recall']:.4f}" for c in CATEGORIES],
                             r["pos_hoc_no_teste"]["fp_para_zero_fn"],
                             f"{r['pos_hoc_no_teste']['recall_com_especificidade_95']:.4f}"])
    figures(dest, results, rows, ranking, y_true, classes, bundles, ids)
    return dest


def figures(dest, results, rows, ranking, y_true, classes, bundles, ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c2c2c2", "axes.labelcolor": "#191919",
                         "xtick.color": "#666666", "ytick.color": "#666666", "grid.color": "#e7e7e7",
                         "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff"})
    fig_dir = dest / "figures"
    fig_dir.mkdir()
    order = list(FINAL_MODELS)
    color = dict(zip(order, COLORS))
    scores_by_model = {m: np.array([r["anomaly_score"] for r in rows if r["modelo"] == m]) for m in order}
    defect = y_true == -1

    # 1. Curvas ROC
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for m in order:
        fpr, tpr, _ = roc_curve(defect, scores_by_model[m])
        ax.plot(fpr, tpr, lw=2, color=color[m], label=f"{results[m]['rotulo']} (AUC {results[m]['auc']:.3f})")
        ax.plot(results[m]["fp"] / (results[m]["fp"] + results[m]["tn"]), results[m]["recall_defeitos"],
                "o", ms=8, color=color[m], mec="#ffffff", mew=2)
    ax.plot([0, 1], [0, 1], ls="--", color="#c2c2c2", lw=1)
    ax.set(xlabel="Taxa de falsos positivos (boas rejeitadas)", ylabel="Recall de defeitos",
           title="Curvas ROC no teste (ponto = limiar calibrado)", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(True, lw=.5)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # 2. FN e FP por modelo (dois painéis, um eixo cada)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    labels = [results[m]["rotulo"] for m in ranking]
    for ax, key, title, total in ((axes[0], "fn", "Falsos negativos (defeitos aprovados)", 119),
                                  (axes[1], "fp", "Falsos positivos (boas rejeitadas)", 41)):
        values = [results[m][key] for m in ranking]
        bars = ax.barh(labels, values, color=[color[m] for m in ranking], height=.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + total * .01, bar.get_y() + bar.get_height() / 2, f"{v} / {total}",
                    va="center", fontsize=9, color="#191919")
        ax.set_title(title)
        ax.set_xlim(0, max(values) * 1.3 + 1)
        ax.grid(True, axis="x", lw=.5)
    axes[0].invert_yaxis()
    fig.suptitle("Erros no limiar calibrado (ordenado pelo critério: menos falsos negativos)")
    fig.tight_layout()
    fig.savefig(fig_dir / "errors_by_model.png", dpi=150)
    plt.close(fig)

    # 3. Recall por categoria de defeito (mapa de calor sequencial)
    matrix = np.array([[results[m]["por_categoria"][c]["recall"] for c in CATEGORIES] for m in ranking])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=20, ha="right")
    ax.set_yticks(range(len(ranking)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            n = results[ranking[i]]["por_categoria"][CATEGORIES[j]]
            ax.text(j, i, f"{n['detectados']}/{n['n']}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if matrix[i, j] > .55 else "#191919")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label="Recall")
    ax.set_title("Defeitos detectados por categoria (teste)")
    fig.tight_layout()
    fig.savefig(fig_dir / "recall_by_category.png", dpi=150)
    plt.close(fig)

    # 4. Matrizes de confusão
    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 3.2))
    for ax, m in zip(axes, ranking):
        r = results[m]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                        color="#ffffff" if cm[i, j] / cm[i].sum() > .55 else "#191919")
        ax.set_xticks([0, 1], ["bom", "defeito"])
        ax.set_yticks([0, 1], ["bom", "defeito"])
        ax.set_xlabel("Predito")
        ax.set_title(r["rotulo"], fontsize=9)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # 5. Distribuição dos scores normalizados pelo limiar
    cats = ("good",) + CATEGORIES
    fig, axes = plt.subplots(len(order), 1, figsize=(8, 1.55 * len(order)), sharex=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, ranking):
        s = scores_by_model[m]
        thr = results[m]["limiar"]
        # Diferença relativa ao limiar, escalada pelo desvio das boas, para comparar modelos.
        good_std = s[~defect].std() or 1.0
        z = (s - thr) / good_std
        for i, c in enumerate(cats):
            mask = np.array(classes) == c
            ax.scatter(z[mask], i + rng.uniform(-.18, .18, mask.sum()), s=12,
                       color="#666666" if c == "good" else color[m], alpha=.8, lw=0)
        ax.axvline(0, color="#191919", lw=1, ls="--")
        ax.set_yticks(range(len(cats)), cats, fontsize=7)
        ax.set_title(f"{results[m]['rotulo']}: (score - limiar) / desvio das boas", fontsize=8.5, loc="left")
        ax.grid(True, axis="x", lw=.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "score_distributions.png", dpi=150)
    plt.close(fig)



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--confirm-test-evaluation", action="store_true")
    args = parser.parse_args(argv)
    try:
        dest = evaluate(args.study, args.out, confirm=args.confirm_test_evaluation)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Avaliação interrompida: {exc}\n")
    print((dest / "summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
