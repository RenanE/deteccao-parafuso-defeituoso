"""Verificação de robustez pós-avaliação (descritiva, não altera modelos nem limiares).

As boas de teste do MVTec screw são mais escuras que todas as demais imagens
(boas de treino, boas reservadas e defeituosas de teste). Este script mede esse
viés de brilho e recalcula a AUC de cada modelo comparando os defeitos de
teste com as 64 boas reservadas (mesma distribuição de brilho do treino), usando
os scores já salvos na calibração final e na avaliação do teste.

    python -m src.robustness_check --study results/stage3/study_v1 \
        --evaluation results/stage8/test_evaluation_v1 --out results/stage8/robustness_v1
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

from .dataset import ler_imagem
from .feature_experiments import save_json
from .final_evaluation import FINAL_MODELS


def brilho(path):
    return float(cv2.cvtColor(ler_imagem(Path(path)), cv2.COLOR_BGR2GRAY).mean())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    outer = json.loads((args.study / "outer_manifest.json").read_text())
    rows = list(csv.DictReader((args.evaluation / "predictions.csv").open(encoding="utf-8")))
    classes = {r["caminho"]: r["classe"] for r in rows}

    groups = {"boas_ajuste": outer["fit_ids"], "boas_reservadas": outer["calibration_ids"],
              "boas_teste": [p for p in outer["test_ids"] if classes[p] == "good"],
              "defeitos_teste": [p for p in outer["test_ids"] if classes[p] != "good"]}
    intensity = {g: np.array([brilho(p) for p in paths]) for g, paths in groups.items()}
    stats = {g: {"n": len(v), "media": float(v.mean()), "desvio": float(v.std()),
                 "min": float(v.min()), "max": float(v.max())} for g, v in intensity.items()}
    auc_brilho = float(roc_auc_score(np.r_[np.zeros(len(intensity["boas_teste"])),
                                           np.ones(len(intensity["defeitos_teste"]))],
                                     np.r_[intensity["boas_teste"], intensity["defeitos_teste"]]))

    per_model = {}
    for name, (folder, family, label) in FINAL_MODELS.items():
        cal = np.array(json.loads((Path(folder) / "final_calibration.json").read_text())["calibration_scores"])
        scores = {r["caminho"]: float(r["anomaly_score"]) for r in rows if r["modelo"] == name}
        good = np.array([scores[p] for p in groups["boas_teste"]])
        defect = np.array([scores[p] for p in groups["defeitos_teste"]])
        def auc(neg, pos):
            return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))
        test_scores = np.r_[good, defect]
        test_int = np.r_[intensity["boas_teste"], intensity["defeitos_teste"]]
        per_model[name] = {"rotulo": label, "familia": family,
                           "auc_teste": auc(good, defect),
                           "auc_reservadas_vs_defeitos": auc(cal, defect),
                           "auc_reservadas_vs_boas_teste": auc(good, cal),
                           "correlacao_score_brilho_teste": float(np.corrcoef(test_scores, test_int)[0, 1])}
    save_json(args.out / "robustness.json", {"brilho_medio_cinza": stats,
              "auc_brilho_boas_teste_vs_defeitos": auc_brilho, "modelos": per_model,
              "nota": "Descritivo; scores e limiares congelados, nenhum modelo foi reajustado."})

    # Figura: histograma de brilho por grupo + AUCs lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = {"boas_ajuste": "#257de0", "boas_reservadas": "#00b894", "boas_teste": "#666666",
              "defeitos_teste": "#f47230"}
    bins = np.linspace(min(v.min() for v in intensity.values()), max(v.max() for v in intensity.values()), 40)
    for g, v in intensity.items():
        axes[0].hist(v, bins=bins, density=True, alpha=.55, color=colors[g], label=f"{g} (n={len(v)})")
    axes[0].set(xlabel="Brilho médio (0-255)", ylabel="Densidade", title="Brilho médio das imagens por grupo")
    axes[0].legend(frameon=False, fontsize=8)
    order = sorted(per_model, key=lambda m: -per_model[m]["auc_reservadas_vs_defeitos"])
    y = np.arange(len(order))
    axes[1].barh(y - .2, [per_model[m]["auc_teste"] for m in order], .38, color="#84b9f3", label="AUC no teste (boas de teste)")
    axes[1].barh(y + .2, [per_model[m]["auc_reservadas_vs_defeitos"] for m in order], .38, color="#1b64be",
                 label="AUC com boas reservadas (sem viés de brilho)")
    axes[1].axvline(.5, color="#c2c2c2", ls="--", lw=1)
    axes[1].set_yticks(y, [per_model[m]["rotulo"] for m in order])
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 1), xlabel="AUC (defeito = positivo)", title="Separação de defeitos")
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "brightness_robustness.png", dpi=150, facecolor="white")
    print(json.dumps({"auc_brilho": auc_brilho, "brilho": {g: round(s["media"], 1) for g, s in stats.items()},
                      **{m: {k: round(v, 3) for k, v in r.items() if isinstance(v, float)} for m, r in per_model.items()}},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
