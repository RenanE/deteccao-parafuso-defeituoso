"""Figuras ilustrativas do pipeline clássico para o relatório e o pôster.

Reutiliza exatamente as funções do pipeline (etapas 1 e 2). Usa apenas imagens
de treino (desenvolvimento) e uma amostra fixa por categoria de teste para ilustração;
não ajusta nem avalia modelos.

    python -m src.report_figures --study results/stage3/study_v1 --out results/stage8/report_figures
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import ler_imagem
from .features import extrair_hog
from .preprocessing import aplicar_clahe, converter_para_cinza, remover_ruido
from .segmentation import aplicar_morfologia, binarizar_otsu, extrair_maior_componente_e_roi

CATS = ("good", "manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top")


def etapas(path):
    cinza = converter_para_cinza(ler_imagem(Path(path)))
    clahe = aplicar_clahe(cinza, clip_limit=2.0, tile_grid_size=(8, 8))
    suavizada = remover_ruido(cinza, metodo="gaussian", ksize=(5, 5))
    _, otsu = binarizar_otsu(suavizada, inverter=True)
    morfo = aplicar_morfologia(otsu, tamanho_abertura=3, tamanho_fechamento=11, preencher_lacunas=True)
    roi, roi_mask, _ = extrair_maior_componente_e_roi(cinza, morfo, alinhar_eixo=True)
    roi_clahe = aplicar_clahe(roi, clip_limit=2.0, tile_grid_size=(8, 8))
    _, hog_img = extrair_hog(roi_clahe, visualize=True)
    return {"Original (cinza)": cinza, "CLAHE": clahe, "Otsu": otsu, "Morfologia": morfo,
            "ROI alinhada + CLAHE": cv2.resize(roi_clahe, (384, 128)), "HOG (8x8)": hog_img}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    outer = json.loads((args.study / "outer_manifest.json").read_text())
    inventory = {d["caminho"]: d["classe"] for d in json.loads((args.study / "inventory.json").read_text())}

    # 1. Etapas do pipeline numa imagem de treino
    steps = etapas(outer["fit_ids"][0])
    fig, axes = plt.subplots(1, len(steps), figsize=(3.2 * len(steps), 3.8))
    for ax, (title, img) in zip(axes, steps.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out / "pipeline_steps.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 2. Uma amostra por categoria de teste e sua ROI alinhada
    examples = [next(p for p in outer["test_ids"] if inventory[p] == c) for c in CATS]
    fig, axes = plt.subplots(2, len(CATS), figsize=(3 * len(CATS), 4.2),
                             gridspec_kw={"height_ratios": [3, 1.2]})
    for j, (cat, path) in enumerate(zip(CATS, examples)):
        s = etapas(path)
        axes[0, j].imshow(s["Original (cinza)"], cmap="gray")
        axes[0, j].set_title(cat, fontsize=11)
        axes[1, j].imshow(s["ROI alinhada + CLAHE"], cmap="gray")
        for ax in axes[:, j]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out / "categories_roi.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 3. ROIs alinhadas de 8 parafusos BONS: mostra a ambiguidade de 180 graus
    fig, axes = plt.subplots(4, 2, figsize=(7, 4.6))
    for ax, path in zip(axes.ravel(), outer["fit_ids"][:8]):
        ax.imshow(etapas(path)["ROI alinhada + CLAHE"], cmap="gray")
        ax.axis("off")
    fig.suptitle("ROIs alinhadas de 8 parafusos bons (desenvolvimento)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "aligned_good_rois.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figuras em {args.out}")


if __name__ == "__main__":
    main()
