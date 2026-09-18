# Archived: superseded by the May 2026 v2 model pipeline

Moved here 2026-09-17. Nothing in this folder produces a current manuscript
figure. Kept for provenance only — full git history is preserved through the
`git mv`, so `git log --follow` still works on every file.

| archived | superseded by |
|---|---|
| `Figure_3/SpliceAI/run_spliceai_all.py` | `Figure_3/SpliceAI/run_spliceai_MAY.py` / `_MAY_gpu.py` |
| `Figure_3/SpliceAI/process_spliceai_04_09_2026.py` | junction extraction is now inline in stage 1 |
| `Figure_3/model_comparison/build_mega_pred_file.py` | `build_mega_MAY_v2.py` |
| `Figure_3/model_comparison/plot_merged_output.py` | `plot_merged_output_MAY_v2.py` |
| `Figure_3/model_comparison/plot_merged_output_MAY.py` | `plot_merged_output_MAY_v2.py` |
| `Figure_3/model_comparison/plot_all_models_comparison.py` | `plot_all_models_comparison_MAY.py` |
| `Figure_3/model_comparison/compare_spliceai_mmsplice_testset.py` | read the April mega; not used in the revision |
| `SI_figures/AlphaGenome/run_alphagenome_predictions.py` | `run_alphagenome_predictions_all_MAY_fullpos.py` |
| `SI_figures/AlphaGenome/run_alphagenome_predictions_all.py` | `run_alphagenome_predictions_all_MAY_fullpos.py` |
| `SI_figures/Pangolin/run_pangolin_all.py` | `run_pangolin_MAY_batched.py` |
| `SI_figures/Pangolin/run_pangolin_all_MAY_fullpos.py` | `run_pangolin_MAY_batched.py` (batching + skip-bug fix) |

`process_spliceai_04_09_2026.py` is the two-stage script whose stage 2 recomputed
`exon_start` from the measurement table while reading a score track built from an
older sequence, joined only on `Reference`. That is what scored ~1,645 tandem-acceptor
variants a few nt off the true junction.
