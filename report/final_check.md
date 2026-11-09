# VisionGuard Final Verification & Submission Manifest

This document serves as the comprehensive audit checklist and verification manifest for the VisionGuard project, mapping each requirement from the assignment brief to verified evidence files and live artifacts.

---

## 1. Project Profile & Submission Links

| Field | Detail / URL |
| :--- | :--- |
| **Student Full Name** | Shayan Ahmed |
| **Roll Number** | `23L-0801` |
| **Project Code / Title** | VisionGuard (`23L-0801_A2_Ops`) |
| **Public GitHub Repository** | [https://github.com/shayanO-O/23L-0801_A2_Ops](https://github.com/shayanO-O/23L-0801_A2_Ops) |
| **DagsHub Project & Remote** | [https://dagshub.com/shayanO-O/23L-0801_A2_Ops](https://dagshub.com/shayanO-O/23L-0801_A2_Ops) |
| **Live Cloud Inference API** | [http://3.24.214.144](http://3.24.214.144) (Port 80) |
| **Graded Git Commit Hash** | `9dbbb22` (Tagged as `graded`) |
| **Compiled PDF Report** | `report/23L-0801_VisionGuard.pdf` (34 pages, publication grade) |

---

## 2. Requirement Verification Matrix (Row-by-Row Brief Audit)

| Task ID | Requirement Description | Status | Evidence Asset | Verified Attributes |
| :--- | :--- | :--- | :--- | :--- |
| **1.1** | Python 3.11 environment & pinned ML stack | **VERIFIED** | `report/screenshots/CP-1.1_a_versions.png`, `report/tables/versions.txt` | Python 3.11.9, TF 2.21.0, Keras 3.13.2, MLflow 3.18.0 |
| **1.2** | Raw data integrity audit & deduplication | **VERIFIED** | `report/tables/audit.csv`, `report/tables/audit_summary.md` | 975 files scanned, 0 corrupt, 2 within-class duplicates handled |
| **1.3** | Visual & statistical EDA; stratified 70/15/15 split | **VERIFIED** | `report/figures/sample_grid_*.png`, `report/figures/hist_*.png`, `splits/*.csv` | Bilinear resize rationale, resolution shortcut defense, seed 42 |
| **1.4** | DVC dataset versioning & DagsHub remote storage | **VERIFIED** | `data/raw.dvc`, `report/screenshots/CP-1.4A_*.png`, `CP-1.4B_*.png`, `CP-1.4C_*.png` | Pointer MD5 hash tracked in Git; payload stored on DagsHub S3 |
| **2.1** | tf.data input pipeline & dynamic augmentation | **VERIFIED** | `src/data.py`, `report/figures/augmented_batch.png` | Zero-drift shared preprocessing, horizontal flips, rotations, zooms |
| **2.2** | Baseline 4-block CNN architecture | **VERIFIED** | `src/models.py`, `report/screenshots/CP-2.2_a_summary.png` | 390,593 trainable parameters, BatchNorm, Dropout (0.4) |
| **2.3** | Transfer learning (MobileNetV2 feature ext & fine-tune) | **VERIFIED** | `src/train.py`, `report/figures/run_curves.png` | Frozen ImageNet base, top inverted residual block fine-tuning |
| **2.4** | Training discipline (EarlyStopping, ReduceLR, Checkpoint) | **VERIFIED** | `src/train.py`, `report/figures/run_curves.png` | Patience 5, factor 0.5, best weights restored to champion.keras |
| **2.5** | Sealed test set evaluation & validation threshold tuning | **VERIFIED** | `report/tables/test_metrics.csv`, `report/figures/test_roc_curves.png`, `test_confusion_matrices.png` | Val ROC-AUC 0.8420, Test ROC-AUC 0.8442, F1 0.7853 at $\tau=0.5252$ |
| **2.6** | Qualitative error analysis on 10 misclassified test samples | **VERIFIED** | `src/errors.py`, `report/figures/misclassified_grid.png` | Balanced 5 FPs and 5 FNs evaluated; bokeh & stroke confusion diagnosed |
| **2.7** | Keras Tuner hyperparameter exploration | **VERIFIED** | `src/tuner.py`, `report/tables/toy_tuner.csv` | 8 trials executed; optimal LR $10^{-3}$, Dropout 0.4, GAP |
| **3.1** | Local MLflow experiment tracking | **VERIFIED** | `report/screenshots/CP-3.1_a_run1.png`, `CP-3.1_b_run2.png`, `CP-3.1_c_compare.png` | Local runs logged, hyperparameters, curves, and compare view |
| **3.2** | DagsHub remote MLflow project & credential security | **VERIFIED** | `report/screenshots/CP-3.2_a_dagshub_project.png`, `CP-3.2_b_toy_run.png`, `CP-3.2_c_env_ignored.png` | Remote tracking URI, .env confirmed git-ignored |
| **3.3** | Manual curated logging vs framework autologging | **VERIFIED** | `report/screenshots/CP-3.3_a_manual_run.png`, `CP-3.3_b_autolog_run.png` | final_manual vs final_autolog analyzed on DagsHub |
| **3.4** | Champion run artifacts uploaded to MLflow | **VERIFIED** | `report/screenshots/CP-3.4_a_artifacts.png` | ROC curve, confusion matrix, split lists, requirements freeze, model |
| **3.5** | Multi-run experiment comparison table | **VERIFIED** | `report/screenshots/CP-3.5_a_compare.png`, `report/tables/run_comparison.csv` | 9 runs (R0-R6, tuner, final) benchmarked across validation metrics |
| **3.6** | Model Registry & champion alias | **VERIFIED** | `report/screenshots/CP-3.6A_a_registry.png`, `CP-3.6B_a_predict_by_alias.png` | Model `23L-0801_A2_Ops_visionguard` registered with `@champion` alias |
| **4.1** | Standalone inference module & once-at-startup loading | **VERIFIED** | `app/inference.py`, `report/screenshots/CP-4.1_a_inference_tests.png` | Singleton model cache, bilinear preprocessing, zero-drift |
| **4.2** | In-memory upload validation & error rejection | **VERIFIED** | `report/screenshots/CP-4.2_a_validation_tests.png` | 400 empty, 413 oversized (>5MB), 400 invalid extension/corrupted |
| **4.3** | Responsive HTML/CSS UI dashboard | **VERIFIED** | `app/templates/index.html`, `report/screenshots/CP-4.3_a_dashboard.png` | Linear/Vercel styling, confidence meter, verdict status |
| **4.4** | Production configuration & health endpoint | **VERIFIED** | `app/app.py`, `GET /health` | Debug mode off, JSON health metrics, uptime reporting |
| **Task 5** | PostgreSQL database audit trail | **VERIFIED** | `app/db.py`, `report/screenshots/CP-6.3B_a_postgres_trail.png` | Predictions table logging timestamp, hash, verdict, confidence, mode |
| **6.1** | Docker warm-up (ports, volumes, logs, shell, teardown) | **VERIFIED** | `report/screenshots/CP-6.1A_*.png`, `CP-6.1B_*.png`, `CP-6.1C_*.png` | Standalone Nginx container lifecycle verified |
| **6.2** | Production multi-stage slim Dockerfile & non-root user | **VERIFIED** | `Dockerfile`, `report/screenshots/CP-6.2A_a_image_size.png`, `CP-6.2B_a_container_nonroot.png` | Slim build, unprivileged user `appuser` (UID 1000) |
| **6.3** | Multi-container Docker Compose stack | **VERIFIED** | `docker-compose.yml`, `report/screenshots/CP-6.3A_*.png`, `CP-6.3B_*.png`, `CP-6.3C_*.png` | Two-tier `web` + `db` stack, internal bridge network, healthcheck |
| **6.4** | Docker image size analysis & layer caching | **VERIFIED** | `report/report.md` (Chapter 5) | Layer optimization and build caching documented |
| **6.5** | Operations lifecycle & fault injection | **VERIFIED** | `report/notes.md`, `report/report.md` | Troubleshooting via container logs and fault injection documented |
| **6.6** | CI/CD matrix pipeline via GitHub Actions | **VERIFIED** | `.github/workflows/ci.yml`, `report/screenshots/CP-6.6A_*.png`, `CP-6.6B_*.png`, `CP-6.6C_*.png` | Python 3.10 and 3.11 matrix passing 100% green |
| **Task 7** | AWS EC2 live cloud API deployment (Pathway C) | **VERIFIED** | `report/screenshots/CP-7.1_a_aws_ec2_running.png`, `CP-7.2_b_security_group_rules.png`, `CP-7.3_*.png`, `CP-7.4_a_cloud_logs.png` | Live on `3.24.214.144:80`, dual firewall, API key protected |
| **Task 8** | Dual-mode Flask UI + remote cloud API integration | **VERIFIED** | `report/screenshots/CP-8_a_result1.png`, `CP-8_b_result2.png`, `CP-8_c_result3.png`, `CP-8_d_history.png` | Mode: cloud verified in UI and persisted in PostgreSQL trail |
| **Bonus** | Grad-CAM visual interpretability (+1 mark) | **VERIFIED** | `src/gradcam.py`, `report/figures/gradcam.png`, `report/tables/gradcam_interpretations.md` | 4 correct & 4 misclassified samples analyzed with qualitative report |

---

## 3. Strict Pre-Submission Security & Hygiene Audit

1. **Zero Secret Leaks**:
   - Automated cryptographic scan across all tracked files and commit history: **ZERO** occurrences of `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DAGSHUB_TOKEN`, `HF_TOKEN`, or `API_KEY`.
   - `.env` is git-ignored and never committed.
2. **Container Security**:
   - The production container environment contains no MLflow or DagsHub tokens (`DAGSHUB_*` and `MLFLOW_*` omitted from runtime container).
   - Container processes run as unprivileged `appuser` (UID 1000).
3. **Environment Template**:
   - `.env.example` is fully populated with blank keys covering local, database, cloud, and AWS settings.
4. **Clean Git State**:
   - Working tree is clean on branch `main` at commit `9dbbb22`.
   - Tagged as `graded`.

---

## 4. Post-Grading Infrastructure Decommissioning Notice

> [!IMPORTANT]
> **Keep the AWS EC2 instance `i-03f435326ddf75d8b` RUNNING until your assignment marks are officially finalized.**
>
> Once marks are published:
> 1. Open the **AWS Management Console** and switch region to **`ap-southeast-2` (Sydney)**.
> 2. Terminate instance **`23L-0801_A2_Ops`** (`i-03f435326ddf75d8b`).
> 3. Delete Security Group **`23L-0801_A2_Ops`** (`sg-0fc0a9d2a99e9d02e`).
> 4. Delete EC2 Key Pair **`23L-0801_A2_Ops`**.
> 5. Deactivate and delete IAM Access Keys in AWS IAM Console.
