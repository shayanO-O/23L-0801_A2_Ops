# VisionGuard Evidence Log

This log records every verified checkpoint and evidence asset, connecting it directly to report requirements.

| Checkpoint ID | Artifact File | What It Proves | Target Report Section |
| :--- | :--- | :--- | :--- |
| CP-1.1 | `report/screenshots/CP-1.1_a_versions.png` | Python 3.11 environment, exact pinned library versions, GPU check | Task 1: Data Understanding and Versioning (1.1) |
| CP-1.4A | `report/screenshots/CP-1.4A_a_toy_commits.png`, `report/screenshots/CP-1.4A_b_toy_pushpull.png` | DVC local versioning, git rollback across data versions, remote push/pull restoration | Task 1: Data Understanding and Versioning (1.4) |
| CP-1.4B | `report/screenshots/CP-1.4B_a_toy_dagshub.png` | DagsHub remote toy repository with Git commits and DVC-tracked dataset pointer | Task 1: Data Understanding and Versioning (1.4) |
| CP-1.4C | `report/screenshots/CP-1.4C_a_pointer.png`, `report/screenshots/CP-1.4C_b_dagshub_data.png` | DVC pointer file contents, in-sync status, and DagsHub web UI showing data/raw tracked via DVC | Task 1: Data Understanding and Versioning (1.4) |
| CP-3.1 | `report/screenshots/CP-3.1_a_run1.png`, `report/screenshots/CP-3.1_b_run2.png`, `report/screenshots/CP-3.1_c_compare.png` | Local MLflow tracking: run 1, run 2, and compare view | Phase 2 / Task 3: Experiment Tracking (3.1) |
| CP-3.2 | `report/screenshots/CP-3.2_a_dagshub_project.png`, `report/screenshots/CP-3.2_b_toy_run.png`, `report/screenshots/CP-3.2_c_env_ignored.png` | DagsHub remote MLflow project, toy run in experiments, git check-ignore proof for .env | Phase 2 / Task 3: Experiment Tracking (3.2) |
| CP-2.2 | `report/screenshots/CP-2.2_a_summary.png`, `report/tables/baseline_summary.csv` | Full baseline CNN model summary with per-layer output shapes and parameter totals (390,593 params) | Task 2: Model Code and Training (2.2) |
| CP-6.1A | `report/screenshots/CP-6.1A_a_browser_update.png`, `report/screenshots/CP-6.1A_b_ps.png` | Docker warm-up: live volume mount update without container restart and running container status | Phase 4 / Task 6: Containerization (6.1) |
| CP-6.1B | `report/screenshots/CP-6.1B_a_logs.png`, `report/screenshots/CP-6.1B_b_shell.png` | Docker container inspection: HTTP access logs and interactive container shell execution | Phase 4 / Task 6: Containerization (6.1) |
| CP-6.1C | `report/screenshots/CP-6.1C_a_cleanup.png` | Docker container teardown: stopped, removed, and port 8085 confirmed free | Phase 4 / Task 6: Containerization (6.1) |
| CP-6.6A | `report/screenshots/CP-6.6A_a_toy_red.png` | GitHub Actions CI toy lab: failing red matrix run caught by pytest | Phase 4 / Task 6: CI/CD Pipeline (6.6) |
| CP-6.6B | `report/screenshots/CP-6.6B_a_toy_green.png` | GitHub Actions CI toy lab: passing green matrix run across Python 3.10 and 3.11 | Phase 4 / Task 6: CI/CD Pipeline (6.6) |
| CP-3.5 | `report/screenshots/CP-3.5_a_compare.png`, `report/tables/run_comparison.csv`, `report/figures/run_comparison.png` | DagsHub MLflow experiments comparison table with all 9 runs and metric columns | Phase 5 / Task 3: Experiment Tracking (3.5) |
| CP-3.3 | `report/screenshots/CP-3.3_a_manual_run.png`, `report/screenshots/CP-3.3_b_autolog_run.png` | DagsHub run pages comparing manual callback logging vs automatic framework autologging | Phase 5 / Task 3: Experiment Tracking (3.3) |
