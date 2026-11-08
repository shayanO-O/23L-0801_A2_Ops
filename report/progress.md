# VisionGuard Progress Trace Table

| Sub-task | Status | Evidence file(s) | Notes |
| :--- | :--- | :--- | :--- |
| 1.1 | Done | `report/tables/versions.txt`, `report/screenshots/CP-1.1_a_versions.png` | Environment setup verified: Python 3.11, pinned packages, GPU check |
| 1.2 | Done | `report/tables/audit.csv`, `report/tables/audit_summary.md` | Dataset audit completed: 975 files, 0 corrupt, 2 duplicates, 0 conflicts |
| 1.3 | Done | `report/figures/sample_grid_*.png`, `report/figures/hist_*.png`, `report/figures/eda_interpretations.md` | Visual EDA completed: samples, dimensional and size distributions, shortcuts analysis |
| 1.4 | Done | `report/screenshots/CP-1.4A_*.png`, `report/screenshots/CP-1.4B_*.png`, `report/screenshots/CP-1.4C_*.png`, `data/raw.dvc` | DVC toy workflow & dataset tracking on DagsHub |
| 2.1 | Done | `report/figures/augmented_batch.png` | tf.data input pipeline, decoding, resizing, augmentation |
| 2.2 | Done | `report/screenshots/CP-2.2_a_summary.png`, `report/tables/baseline_summary.csv` | Baseline CNN architecture, layer table, summary |
| 2.3 | Done | `report/figures/curves_*.png`, `report/tables/trainable_layers_*.csv` | Transfer learning MobileNetV2, feature extraction & fine-tuning |
| 2.4 | Done | `report/figures/training_curves_discipline.png` | Seeds, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| 2.5 | Done | `report/tables/test_metrics.csv`, `report/figures/roc_curve.png`, `report/figures/confusion_matrix.png` | Test set evaluation (once), threshold tuning on val (tau = 0.5252) |
| 2.6 | Done | `report/figures/misclassified_grid.png` | Error analysis of 10 misclassified test images |
| 2.7 | Done | `report/tables/toy_tuner.csv`, `report/tables/tuner_trials.csv` | Keras Tuner 8-trial hyperparameter search executed |
| 3.1 | Done | `report/screenshots/CP-3.1_*.png` | MLflow local toy runs, parameters, metrics, comparison |
| 3.2 | Done | `report/screenshots/CP-3.2_*.png` | DagsHub MLflow remote tracking, env verification |
| 3.3 | Done | `report/screenshots/CP-3.3_*.png` | Manual callback vs autolog comparison on DagsHub |
| 3.4 | Done | `report/screenshots/CP-3.4_a_artifacts.png`, `samples/` | Logged artifacts: ROC, confusion matrix, splits, freeze, model |
| 3.5 | Done | `report/screenshots/CP-3.5_a_compare.png`, `report/tables/run_comparison.csv` | Experiment comparison table across 9 runs (R0-R6, tuner, final) |
| 3.6 | Done | `report/screenshots/CP-3.6A_*.png`, `report/screenshots/CP-3.6B_*.png` | Model registry on DagsHub, champion alias, prediction by alias |
| 4.1 | Done | `report/screenshots/CP-4.1_a_inference_tests.png` | Standalone inference module, once-at-startup loading |
| 4.2 | Done | `report/screenshots/CP-4.2_a_validation_tests.png` | Routes, in-memory validation, rejection handling (bad type, size, corrupt) |
| 4.3 | Done | `report/screenshots/CP-4.3_a_dashboard.png` | HTML/CSS UI: upload form, verdict & confidence display, history link |
| 4.4 | Done | `app/config.py`, `app/app.py` | Config via env vars, Flask server execution, debug mode off |
| Task 5 | Done | `app/models.py`, `report/screenshots/CP-6.3B_a_postgres_trail.png` | Database trail (SQLite/Postgres), predictions logged & queryable |
| 6.1 | Done | `report/screenshots/CP-6.1A_*.png`, `report/screenshots/CP-6.1B_*.png`, `report/screenshots/CP-6.1C_*.png` | Docker warm-up: run, port, volume, logs, shell, cleanup |
| 6.2 | Done | `report/screenshots/CP-6.2A_a_image_size.png`, `report/screenshots/CP-6.2B_a_container_nonroot.png` | Slim Dockerfile, non-root user (UID 1000), healthcheck, tensorflow-cpu |
| 6.3 | Done | `report/screenshots/CP-6.3A_*.png`, `report/screenshots/CP-6.3B_*.png`, `report/screenshots/CP-6.3C_*.png` | Multi-container stack (web + db), internal networking, healthcheck |
| 6.4 | Done | `report/tables/docker_image_sizes.txt` | Image size analysis, layer caching, optimization before/after |
| 6.5 | Done | `report/notes.md` | Operations lifecycle, fault injection, troubleshooting via logs |
| 6.6 | Done | `report/screenshots/CP-6.6A_*.png`, `report/screenshots/CP-6.6B_*.png`, `report/screenshots/CP-6.6C_*.png` | GitHub Actions CI: toy matrix & main project pipeline (Python 3.10 & 3.11 green) |
| Task 7 | Done | `report/screenshots/CP-7.*.png`, `src/aws_deploy.py`, `src/aws_setup_host.py` | AWS EC2 t3.small live deployment in ap-southeast-2, Docker container on port 80, API key protected |
| Task 8 | In Progress | `report/screenshots/CP-8_*.png` | Flask localhost integration with cloud API, history verification |
