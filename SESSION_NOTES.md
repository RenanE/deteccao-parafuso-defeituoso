# Session notes (Claude Code, 2026-11-25)

Summary of the working session that finished the project, to continue later.
Full technical details: `ALTERACOES.md` (what changed and why) and `README.md`
(section "Etapas 6 a 9").

## What was done

1. Environment: Linux venv (the repo `.venv` is Windows-only). Dataset copied
   byte for byte to `/home/gabrieladc/Downloads/data` because the signed manifests
   store that absolute path. `src/*.py` and `tests/*.py` converted CRLF to LF
   (identical to git) so the signed source hashes match.
2. Stage 6: LOF added (`src/lof.py`, `src/lof_experiments.py`).
3. Final calibration of the global models on the 64 reserved good images.
4. Stage 8: global models on the full test set. Result: near chance
   (AUC 0.44 to 0.63). Found a dataset bias: test good images are darker.
5. Neural networks (PatchCore, autoencoder) were built, then moved to
   `neural_networks/` when the project scope was set to classical only
   (SVM, LOF, Isolation Forest).
6. Stage 9: diagnosis (global HOG has no signal, AUC 0.48) and new local
   descriptors per patch on the aligned ROI (`src/local_features.py`).
   Test split into selection / holdout before experiments (`src/defect_split.py`).
   Search of 168 combinations on the selection half; single evaluation on the holdout.
7. Holdout result (61 defects, 21 good): LOF local misses 2 defects (was 53),
   3 false positives, AUC 0.958 (was 0.561). SVM misses 11, Isolation Forest 19.
8. Deliverables: classical report `docs/relatorio/index.html` (also published at
   https://claude.ai/artifact/8vJMP8z6tKzTUq9UXNYwRT, private), A3 poster
   `docs/poster/poster_a3.pdf`, notebook `notebooks/04_avaliacao_classica.ipynb`.
   109 tests pass.

## Open question discussed at the end

The test set had been seen in stage 8, which may bias the result upward
(mainly the idea of going local, plus selecting the best of 168 configs).
Holdout 95% intervals: recall 59/61 between 89% and 99%; specificity 18/21
between 65% and 95%.

## Next step (agreed plan, not run yet)

- Option 2: repeat the whole stage 9 procedure on 10 to 20 random selection/holdout
  splits (different seeds) and report the spread of holdout results.
- Option 4: evaluate a fixed default config chosen before the search
  (LOF k=20, patch 32, all descriptors, neighbors 1) on the full test set,
  to show the result does not depend on tuning.
- Then add the results to README, `ALTERACOES.md`, report and poster.
- Option 3 (synthetic defects for selection) is more rigorous but needs new code.

## Running things again (Linux)

```bash
export DATASET_DIR=/home/gabrieladc/Downloads/data TRAIN_FLAT_IS_GOOD=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest discover -s tests
python -m src.stage9_experiments select --out results/stage9/selection_v2 --resume
```

Long runs should be started detached (`setsid nohup ... &`) so they survive the
session ending; `select` writes `runs.jsonl` and can be resumed with `--resume`.
