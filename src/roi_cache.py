"""Etapa 9: cache das ROIs alinhadas (mesmo pré-processamento/segmentação das etapas 1-2).

Salva, para as 480 imagens, a ROI em cinza e a máscara redimensionadas para
384x128 (a mesma resolução usada pelo HOG), além do contorno superior/inferior.
Nenhum parâmetro do pipeline original é alterado.

    python -m src.roi_cache --study results/stage3/study_v1 --out results/stage9/roi_cache_v1.npz
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed

from .dataset import ler_imagem
from .feature_experiments import file_hash
from .preprocessing import converter_para_cinza, remover_ruido
from .segmentation import aplicar_morfologia, binarizar_otsu, extrair_maior_componente_e_roi

ROI_W, ROI_H = 384, 128


def roi_de(path):
    cinza = converter_para_cinza(ler_imagem(Path(path)))
    suavizada = remover_ruido(cinza, metodo="gaussian", ksize=(5, 5))
    _, otsu = binarizar_otsu(suavizada, inverter=True)
    mascara = aplicar_morfologia(otsu, tamanho_abertura=3, tamanho_fechamento=11, preencher_lacunas=True)
    roi, roi_mask, _ = extrair_maior_componente_e_roi(cinza, mascara, alinhar_eixo=True)
    if roi is None:
        raise ValueError(f"ROI não extraída: {path}")
    roi = cv2.resize(roi, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA)
    roi_mask = (cv2.resize(roi_mask, (ROI_W, ROI_H), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return roi, roi_mask, float(cinza.mean())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"Destino existente: {args.out}")
    inventory = {d["caminho"]: d for d in json.loads((args.study / "inventory.json").read_text())}
    outer = json.loads((args.study / "outer_manifest.json").read_text())
    ids = outer["fit_ids"] + outer["calibration_ids"] + outer["test_ids"]
    for p in ids:
        if file_hash(p) != inventory[p]["sha256"]:
            raise SystemExit(f"Imagem alterada: {p}")
    out = Parallel(n_jobs=args.jobs, verbose=0)(delayed(roi_de)(p) for p in ids)
    rois = np.stack([o[0] for o in out])
    masks = np.stack([o[1] for o in out])
    brightness = np.array([o[2] for o in out])
    groups = np.array(["fit"] * len(outer["fit_ids"]) + ["calibration"] * len(outer["calibration_ids"])
                      + ["test"] * len(outer["test_ids"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, ids=np.asarray(ids), groups=groups, rois=rois, masks=masks,
                            brightness=brightness, classes=np.asarray([inventory[p]["classe"] for p in ids]),
                            outer_manifest_hash=outer["manifest_hash"])
    print(f"{len(ids)} ROIs salvas em {args.out}: {rois.shape}")


if __name__ == "__main__":
    main()
