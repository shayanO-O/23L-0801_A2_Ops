# VisionGuard Evidence Log

This log records every verified checkpoint and evidence asset, connecting it directly to report requirements.

| Checkpoint ID | Artifact File | What It Proves | Target Report Section |
| :--- | :--- | :--- | :--- |
| CP-1.1 | `report/screenshots/CP-1.1_a_versions.png` | Python 3.11 environment, exact pinned library versions, GPU check | Task 1: Data Understanding and Versioning (1.1) |
| CP-1.4A | `report/screenshots/CP-1.4A_a_toy_commits.png`, `report/screenshots/CP-1.4A_b_toy_pushpull.png` | DVC local versioning, git rollback across data versions, remote push/pull restoration | Task 1: Data Understanding and Versioning (1.4) |
| CP-1.4B | `report/screenshots/CP-1.4B_a_toy_dagshub.png` | DagsHub remote toy repository with Git commits and DVC-tracked dataset pointer | Task 1: Data Understanding and Versioning (1.4) |
| CP-1.4C | `report/screenshots/CP-1.4C_a_pointer.png`, `report/screenshots/CP-1.4C_b_dagshub_data.png` | DVC pointer file contents, in-sync status, and DagsHub web UI showing data/raw tracked via DVC | Task 1: Data Understanding and Versioning (1.4) |
| CP-2.2 | `report/screenshots/CP-2.2_a_summary.png`, `report/tables/baseline_summary.csv` | Full baseline CNN model summary with per-layer output shapes and parameter totals (390,593 params) | Task 2: Model Code and Training (2.2) |
| CP-3.1 | `report/screenshots/CP-3.1_a_run1.png`, `report/screenshots/CP-3.1_b_run2.png`, `report/screenshots/CP-3.1_c_compare.png` | Local MLflow tracking: run 1, run 2, and compare view | Phase 2 / Task 3: Experiment Tracking (3.1) |
| CP-3.2 | `report/screenshots/CP-3.2_a_dagshub_project.png`, `report/screenshots/CP-3.2_b_toy_run.png`, `report/screenshots/CP-3.2_c_env_ignored.png` | DagsHub remote MLflow project, toy run in experiments, git check-ignore proof for .env | Phase 2 / Task 3: Experiment Tracking (3.2) |
| CP-3.3 | `report/screenshots/CP-3.3_a_manual_run.png`, `report/screenshots/CP-3.3_b_autolog_run.png` | DagsHub run pages comparing manual callback logging vs automatic framework autologging | Phase 5 / Task 3: Experiment Tracking (3.3) |
| CP-3.4 | `report/screenshots/CP-3.4_a_artifacts.png` | DagsHub artifact repository showing confusion matrix, ROC curve, classification report, and splits | Phase 5 / Task 3: Artifact Logging (3.4) |
| CP-3.5 | `report/screenshots/CP-3.5_a_compare.png`, `report/tables/run_comparison.csv`, `report/figures/run_comparison.png` | DagsHub MLflow experiments comparison table with all 9 runs and metric columns | Phase 5 / Task 3: Experiment Tracking (3.5) |
| CP-3.6A | `report/screenshots/CP-3.6A_a_registry.png` | DagsHub Model Registry showing registered model `23L-0801_A2_Ops_visionguard` and `@champion` alias | Phase 5 / Task 3: Model Registry (3.6) |
| CP-3.6B | `report/screenshots/CP-3.6B_a_predict_by_alias.png` | Standalone script loading champion model by alias `@champion` from DagsHub and running inference | Phase 5 / Task 3: Model Registry (3.6) |
| CP-4.1 | `report/screenshots/CP-4.1_a_inference_tests.png` | Standalone inference module unit tests with once-at-startup model loading passing | Phase 6 / Task 4: Standalone Inference (4.1) |
| CP-4.2 | `report/screenshots/CP-4.2_a_validation_tests.png` | In-memory upload validation tests (MIME type, 5MB limit, corrupt file rejection) passing | Phase 6 / Task 4: Input Validation (4.2) |
| CP-4.3 | `report/screenshots/CP-4.3_a_dashboard.png` | Polished web interface displaying classification verdict, confidence percentage, and preview | Phase 6 / Task 4: UI Dashboard (4.3) |
| CP-6.1A | `report/screenshots/CP-6.1A_a_browser_update.png`, `report/screenshots/CP-6.1A_b_ps.png` | Docker warm-up: live volume mount update without container restart and running container status | Phase 4 / Task 6: Containerization (6.1) |
| CP-6.1B | `report/screenshots/CP-6.1B_a_logs.png`, `report/screenshots/CP-6.1B_b_shell.png` | Docker container inspection: HTTP access logs and interactive container shell execution | Phase 4 / Task 6: Containerization (6.1) |
| CP-6.1C | `report/screenshots/CP-6.1C_a_cleanup.png` | Docker container teardown: stopped, removed, and port 8085 confirmed free | Phase 4 / Task 6: Containerization (6.1) |
| CP-6.2A | `report/screenshots/CP-6.2A_a_image_size.png` | Multi-stage slim Docker image size verification under budget | Phase 7 / Task 6: Production Docker Image (6.2) |
| CP-6.2B | `report/screenshots/CP-6.2B_a_container_nonroot.png` | Container execution under dedicated unprivileged non-root user `appuser` (UID 1000) | Phase 7 / Task 6: Container Security (6.2) |
| CP-6.3A | `report/screenshots/CP-6.3A_a_compose_ps.png` | Multi-container Docker Compose stack running with both `web` and `db` healthy | Phase 7 / Task 6: Multi-Container Stack (6.3) |
| CP-6.3B | `report/screenshots/CP-6.3B_a_postgres_trail.png` | PostgreSQL database audit trail with logged prediction timestamps and metadata | Phase 7 / Task 6: Database Persistence (6.3) |
| CP-6.3C | `report/screenshots/CP-6.3C_a_compose_down.png` | Docker Compose graceful stack teardown and volume preservation | Phase 7 / Task 6: Compose Lifecycle (6.3) |
| CP-6.6A | `report/screenshots/CP-6.6A_a_toy_red.png` | GitHub Actions CI toy lab: failing red matrix run caught by pytest | Phase 4 / Task 6: CI/CD Pipeline (6.6) |
| CP-6.6B | `report/screenshots/CP-6.6B_a_toy_green.png` | GitHub Actions CI toy lab: passing green matrix run across Python 3.10 and 3.11 | Phase 4 / Task 6: CI/CD Pipeline (6.6) |
| CP-6.6C | `report/screenshots/CP-6.6C_a_main_ci_green.png` | GitHub Actions VisionGuard production matrix CI workflow passing across Python 3.10 and 3.11 | Phase 7 / Task 6: CI Pipeline (6.6) |
| CP-7.1 | `report/screenshots/CP-7.1_a_aws_ec2_running.png` | AWS EC2 Console showing instance `23L-0801_A2_Ops` running with Public IPv4 `3.24.214.144` | Phase 9 / Task 7: Cloud Deployment (7.1) |
| CP-7.2_b | `report/screenshots/CP-7.2_b_security_group_rules.png` | AWS EC2 Security Group `23L-0801_A2_Ops` inbound rules: SSH 22 restricted to IP, HTTP 80 open | Phase 9 / Task 7: Cloud Security (7.2) |
| CP-7.3 | `report/screenshots/CP-7.3_a_health.png`, `report/screenshots/CP-7.3_b_predict_real.png`, `report/screenshots/CP-7.3_c_predict_ai.png`, `report/screenshots/CP-7.3_d_no_key_401.png`, `report/screenshots/CP-7.3_e_other_network.png` | Postman execution from phone hotspot: GET /health 200, POST /predict (Real & AI 200), 401 Unauthorized rejection without API key, and mobile network proof | Phase 9 / Task 7: Cloud Verification (7.3) |
| CP-7.4 | `report/screenshots/CP-7.4_a_cloud_logs.png` | EC2 container docker logs showing external and mobile hotspot requests to /health and /predict | Phase 9 / Task 7: Cloud Logs (7.4) |
| CP-8 | `report/screenshots/CP-8_a_result1.png`, `report/screenshots/CP-8_b_result2.png`, `report/screenshots/CP-8_c_result3.png`, `report/screenshots/CP-8_d_history.png` | Dual-mode Flask UI executing remote cloud inference against AWS EC2, showing Real & AI verdict screens with address bar and full PostgreSQL audit trail with Mode: cloud | Phase 10 / Task 8: Flask + Cloud Integration (CP-8) |
| Bonus | `report/figures/gradcam.png`, `report/tables/gradcam_interpretations.md`, `src/gradcam.py` | Grad-CAM gradient-weighted class activation mapping for 4 correct and 4 wrong test predictions with qualitative feature analysis | Phase 11: Optional Bonus (Grad-CAM) |
