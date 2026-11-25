"""Etapa 9: divisão congelada do teste em SELEÇÃO e HOLDOUT.

Motivo: o teste completo já foi consultado na etapa 8. Para buscar melhorias nos
modelos clássicos sem ajustar ao teste, metade dos defeitos de cada categoria
vira conjunto de seleção (escolha de representação/hiperparâmetros) e a outra
metade, junto com metade das boas de teste, fica guardada para um único
relatório final. Durante a seleção, os negativos são as 64 boas reservadas
(mesma distribuição de brilho do treino), nunca as boas de teste.

    python -m src.defect_split --study results/stage3/study_v1 --out results/stage9/defect_split_v1.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .feature_experiments import save_json, signature

SEED = 2026


def make_split(inventory, test_ids, seed=SEED):
    classes = {d["caminho"]: d["classe"] for d in inventory}
    rng = np.random.default_rng(seed)
    selection, holdout = [], []
    for cat in sorted({classes[p] for p in test_ids}):
        ids = sorted(p for p in test_ids if classes[p] == cat)
        order = rng.permutation(len(ids))
        half = len(ids) // 2
        chosen = sorted(ids[i] for i in order[:half])
        selection += chosen
        holdout += sorted(set(ids) - set(chosen))
    return {"seed": seed, "selection_ids": sorted(selection), "holdout_ids": sorted(holdout),
            "classes": {p: classes[p] for p in test_ids},
            "rule": "Estratificado por categoria; floor(n/2) para seleção, restante para holdout. "
                    "Na seleção, só os defeitos são usados (negativos = 64 boas reservadas)."}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"Destino existente: {args.out}")
    inventory = json.loads((args.study / "inventory.json").read_text())
    outer = json.loads((args.study / "outer_manifest.json").read_text())
    split = make_split(inventory, outer["test_ids"])
    split["outer_manifest_hash"] = outer["manifest_hash"]
    split["signature"] = signature(split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out, split)
    for name in ("selection_ids", "holdout_ids"):
        cats = [split["classes"][p] for p in split[name]]
        print(name, {c: cats.count(c) for c in sorted(set(cats))})


if __name__ == "__main__":
    main()
