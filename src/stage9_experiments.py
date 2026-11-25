"""Etapa 9: melhoria dos modelos clássicos (SVM, LOF, Isolation Forest) com descritores locais.

Protocolo (definido antes dos experimentos):
- Ajuste: 256 boas de desenvolvimento (fit_ids). Nada de teste no ajuste.
- Seleção: AUC entre as 64 boas reservadas e os 58 defeitos da metade de SELEÇÃO
  do teste (results/stage9/defect_split_v1.json); desempate pelo recall com
  especificidade de 95% nas reservadas. As boas de teste não entram na seleção.
- Final: limiar = quantil 95% (method="higher") dos scores das 64 reservadas;
  avaliação única no HOLDOUT (21 boas + 61 defeitos de teste).

    python -m src.stage9_experiments select --out results/stage9/selection_v1
    python -m src.stage9_experiments final --selection results/stage9/selection_v1/selection.json \
        --out results/stage9/final_v1
"""

import argparse
import csv
import itertools
import json
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from .feature_experiments import file_hash, save_json, signature
from .local_features import LocalConfig, PatchAnomalyDetector, extrair_lote

CACHE = Path("results/stage9/roi_cache_v1.npz")
SPLIT = Path("results/stage9/defect_split_v1.json")
FEATURE_GRID = [LocalConfig(patch=p, descriptors=d, znorm=True, neighbors=n)
                for p in (24, 32, 48)
                for d in (("hog", "lbp", "gabor", "edges", "shape"), ("lbp", "gabor", "edges", "shape"))
                for n in (1, 2)]
MODEL_GRID = ([("lof", {"n_neighbors": k}) for k in (10, 20, 40)]
              + [("isolation_forest", {"n_estimators": 200, "max_samples": m}) for m in ("auto", 512)]
              + [("svm", {"nu": nu}) for nu in (0.05, 0.1)])
ABLATION = [("hog",), ("lbp",), ("gabor",), ("edges", "shape"), ("lbp", "gabor", "edges", "shape"),
            ("hog", "lbp", "gabor", "edges", "shape")]


def load():
    c = np.load(CACHE)
    split = json.loads(SPLIT.read_text())
    ids = [str(x) for x in c["ids"]]
    groups = c["groups"]
    cls = split["classes"]
    idx = {
        "fit": np.where(groups == "fit")[0], "calibration": np.where(groups == "calibration")[0],
        "sel_defects": np.array([i for i, p in enumerate(ids) if p in set(split["selection_ids"]) and cls[p] != "good"]),
        "sel_goods": np.array([i for i, p in enumerate(ids) if p in set(split["selection_ids"]) and cls[p] == "good"]),
        "hold_defects": np.array([i for i, p in enumerate(ids) if p in set(split["holdout_ids"]) and cls[p] != "good"]),
        "hold_goods": np.array([i for i, p in enumerate(ids) if p in set(split["holdout_ids"]) and cls[p] == "good"]),
    }
    return c, ids, idx, split


def auc(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def recall_at_spec(neg, pos, spec=0.95):
    thr = np.quantile(neg, spec, method="higher")
    return float(np.mean(pos > thr))


def select(out, resume=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=resume)
    if (out / "selection.json").exists():
        raise SystemExit(f"Seleção já concluída: {out}")
    c, ids, idx, split = load()
    rows, features = [], {}
    log_path = out / "runs.jsonl"
    done = {}
    if resume and log_path.exists():
        # Retoma uma execução interrompida: cada ajuste já registrado não é refeito.
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            done.setdefault((r["tag"], r["config"].rsplit("-", 1)[0], r["model"], r["params"]), []).append(r)
        print(f"Retomando: {sum(len(v) for v in done.values())} linhas já registradas", flush=True)
    log = log_path.open("a")

    def run(cfg, model, params, tag):
        key = cfg.key()
        previous = done.get((tag, key.rsplit("-", 1)[0], model, json.dumps(params, sort_keys=True)))
        if previous:
            rows.extend(previous)
            return
        if key not in features:
            features[key] = extrair_lote(c["rois"], c["masks"], cfg)
        F = features[key]
        t = time.perf_counter()
        det = PatchAnomalyDetector(model, params, cfg)
        cal_map, sel_map = det.fit_score(F[idx["fit"]], [F[idx["calibration"]], F[idx["sel_defects"]]])
        out_rows = []
        for agg in ("max", "top3"):
            det.config = LocalConfig(**{**asdict(cfg), "aggregation": agg})
            cal, sel = det.aggregate(cal_map), det.aggregate(sel_map)
            out_rows.append({"tag": tag, "config": det.config.key(), "model": model,
                             "params": json.dumps(params, sort_keys=True), "patch": cfg.patch,
                             "descriptors": "+".join(cfg.descriptors), "neighbors": cfg.neighbors,
                             "aggregation": agg, "dim": int(F.shape[-1]),
                             "auc_sel": auc(cal, sel), "recall95_sel": recall_at_spec(cal, sel),
                             "seconds": time.perf_counter() - t})
        if tag != "grade":
            out_rows = out_rows[:1]
        rows.extend(out_rows)
        for r in out_rows:
            log.write(json.dumps(r) + "\n")
        log.flush()
        r = out_rows[0]
        print(f"[{len(rows)}] {tag} {model} {params} {r['config']}: AUC max={r['auc_sel']:.3f} "
              f"recall95={r['recall95_sel']:.2f}" + (f" | top3 AUC={out_rows[1]['auc_sel']:.3f}" if len(out_rows) > 1 else "")
              + f" ({r['seconds']:.0f}s)", flush=True)

    # 1) Grade principal (agregação max e top3 compartilham o ajuste).
    for cfg, (model, params) in itertools.product(FEATURE_GRID, MODEL_GRID):
        run(cfg, model, params, "grade")
    # 2) Ablação dos descritores (LOF k=20, P=32, vizinhança 1), para o relatório.
    for desc in ABLATION:
        run(LocalConfig(patch=32, descriptors=desc, znorm=True, neighbors=1), "lof", {"n_neighbors": 20}, "ablacao")
    run(LocalConfig(patch=32, znorm=False, neighbors=1), "lof", {"n_neighbors": 20}, "ablacao_sem_znorm")
    run(LocalConfig(patch=32, neighbors=0), "lof", {"n_neighbors": 20}, "ablacao_sem_vizinhos")

    log.close()
    with (out / "runs.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    grid = [r for r in rows if r["tag"] == "grade"]
    selected = {}
    for model in ("svm", "lof", "isolation_forest"):
        cand = [r for r in grid if r["model"] == model]
        best = max(cand, key=lambda r: (round(r["auc_sel"], 6), r["recall95_sel"], -r["dim"]))
        selected[model] = {k: best[k] for k in ("config", "params", "patch", "descriptors", "neighbors",
                                                 "aggregation", "auc_sel", "recall95_sel")}
    selection = {"selected": selected, "rule": "Maior AUC (reservadas vs defeitos de seleção); "
                 "desempate: recall a 95% de especificidade, depois menor dimensão.",
                 "split_signature": split["signature"], "roi_cache_sha256": file_hash(CACHE),
                 "sources": {n: file_hash(Path("src") / n) for n in ("local_features.py", "stage9_experiments.py")},
                 "n_runs": len(rows)}
    selection["signature"] = signature(selection)
    save_json(out / "selection.json", selection)
    print(json.dumps(selected, indent=1))


def config_from(sel):
    return LocalConfig(patch=sel["patch"], descriptors=tuple(sel["descriptors"].split("+")), znorm=True,
                       neighbors=sel["neighbors"], aggregation=sel["aggregation"])


def final(selection_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(Path(selection_path).read_text())
    if signature({k: v for k, v in selection.items() if k != "signature"}) != selection["signature"]:
        raise SystemExit("Seleção alterada.")
    c, ids, idx, split = load()
    if file_hash(CACHE) != selection["roi_cache_sha256"]:
        raise SystemExit("Cache de ROIs alterado.")
    classes = np.array([split["classes"].get(p, "good") for p in ids])
    results, preds, maps = {}, [], {}
    for model, sel in selection["selected"].items():
        cfg = config_from(sel)
        F = extrair_lote(c["rois"], c["masks"], cfg)
        det = PatchAnomalyDetector(model, json.loads(sel["params"]), cfg)
        hold_goods, hold_def = idx["hold_goods"], idx["hold_defects"]
        cats = sorted(set(classes[hold_def]))
        ex = np.r_[hold_goods[:1], [hold_def[classes[hold_def] == cat][0] for cat in cats]].astype(int)
        cal_m, hg_m, hd_m, ex_m = det.fit_score(F[idx["fit"]], [F[idx["calibration"]], F[hold_goods], F[hold_def], F[ex]])
        cal, hg, hd = det.aggregate(cal_m), det.aggregate(hg_m), det.aggregate(hd_m)
        thr = float(np.quantile(cal, 0.95, method="higher"))
        tp, fn = int((hd > thr).sum()), int((hd <= thr).sum())
        fp, tn = int((hg > thr).sum()), int((hg <= thr).sum())
        per_cat = {}
        for cat in sorted(set(classes[idx["hold_defects"]])):
            m = classes[idx["hold_defects"]] == cat
            per_cat[cat] = {"n": int(m.sum()), "detectados": int((hd[m] > thr).sum())}
        results[model] = {"config": sel["config"], "params": sel["params"], "threshold": thr,
                          "calibration_fpr": float(np.mean(cal > thr)),
                          "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                          "recall_defeitos": tp / (tp + fn), "especificidade": tn / (tn + fp),
                          "auc_holdout": auc(hg, hd), "auc_reservadas_vs_defeitos_holdout": auc(cal, hd),
                          "por_categoria": per_cat}
        for i, s in zip(np.r_[idx["hold_goods"], idx["hold_defects"]], np.r_[hg, hd]):
            preds.append({"modelo": model, "caminho": ids[i], "classe": classes[i], "score": float(s),
                          "predito": -1 if s > thr else 1})
        maps[model] = {"ids": [ids[i] for i in ex], "maps": ex_m, "threshold": thr, "patch": cfg.patch}
        print(model, {k: results[model][k] for k in ("fn", "fp", "recall_defeitos", "especificidade", "auc_holdout",
                                                    "auc_reservadas_vs_defeitos_holdout")}, flush=True)
    save_json(out / "holdout_evaluation.json", {"selection_signature": selection["signature"],
                                                 "n_holdout_goods": len(idx["hold_goods"]),
                                                 "n_holdout_defects": len(idx["hold_defects"]), "results": results})
    with (out / "predictions.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0]))
        w.writeheader()
        w.writerows(preds)
    np.savez_compressed(out / "example_maps.npz", **{f"{m}_maps": v["maps"] for m, v in maps.items()},
                        **{f"{m}_ids": np.array(v["ids"]) for m, v in maps.items()},
                        **{f"{m}_threshold": v["threshold"] for m, v in maps.items()},
                        **{f"{m}_patch": v["patch"] for m, v in maps.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true", help="Retoma a partir de runs.jsonl")
    p = sub.add_parser("final")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "select":
        select(args.out, resume=args.resume)
    else:
        final(args.selection, args.out)


if __name__ == "__main__":
    main()
