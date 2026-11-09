# VisionGuard (Assignment 2) - Antigravity Master Prompt

You will complete the whole university assignment "VisionGuard: AI vs Real Image Classification and Deployment" for me, end to end.

The brief is `Assignment_2.pdf` in this workspace. **The brief is the highest authority.** Read it fully first. If this prompt and the brief ever disagree, follow the brief and tell me about the conflict. Read this whole file before doing anything. Do not summarise it away.

---

## 1. Operating Rules

1. **Scope.** Do exactly what the brief asks. No extra features, no advanced techniques, no extra dependencies, files or abstractions. The only optional item is the Grad-CAM bonus (Phase 11) and only if I say yes.
2. **Automate everything you can**: files, installs, code runs, git, `gh`, `dvc`, `docker`, and the `huggingface_hub` Python API. Stop only at (a) a **CHECKPOINT** (screenshot needed) or (b) a **MANUAL STOP** (something only I can do). Both are defined below.
3. **Code style.** Simple, readable, small functions, plain Python. Few comments, only where something is not obvious. **Every comment is in Subject Case**, for example `# Load The Model Once`.
4. **Git.** Commit after each finished sub-task (or small group). Commit message = a few words in Subject Case, for example `Add Data Audit Script`. Push after each phase. What may be committed is fixed by Rule 13.
5. **Secrets.** Only in `.env` (git-ignored), Colab Secrets and Hugging Face Space secrets (for the cloud API key). Docker Compose passes each container only the variables it needs, never the whole `.env`. Never print, echo, log, hard-code or put a token, password or API key in code, notebooks, terminal output you show me, or the report. Keep `.env.example` with names and empty/dummy values.
6. **Naming.** The project name is `<ROLL>_A2_Ops` (roll number collected in MS-0). Define `<SLUG>` = project name lowercased, every character that is not a letter or digit replaced by `-`, prefixed with `vg-` if it would start with a digit. Use these names, and tell me the final values:
   - GitHub repo, DagsHub repo, MLflow experiment, report: `<ROLL>_A2_Ops`.
   - MLflow registered model: `<ROLL>_A2_Ops_visionguard` (the alias stays `champion`).
   - Docker Compose project (`name:`): `<SLUG>`.
   - Hugging Face Space (cloud API): `<SLUG>-visionguard-api` (letters, digits, hyphens). Public URL: `https://<hf_user>-<SLUG>-visionguard-api.hf.space` (lowercase).
   Where another platform forbids the characters, adapt minimally, keep the roll number recognisable, and tell me.
7. **Seed.** `SEED = 42` everywhere (python, numpy, tensorflow, splits).
8. **Versions.** Use the identical TensorFlow/Keras/MLflow versions locally, in Colab and in Docker (Docker uses `tensorflow-cpu` of the same version). Model files must load everywhere. Pin `keras` explicitly in `requirements-train.txt` (so installing Keras Tuner cannot change it) and verify the versions after every install.
9. **Trace table.** Create `report/progress.md` at the start: one row for every sub-task ID in the brief (1.1 to 1.4, 2.1 to 2.7, 3.1 to 3.6, 4.1 to 4.4, Task 5, 6.1 to 6.6, Task 7, Task 8) with columns `Status | Evidence file(s) | Notes`. Update it after every sub-task. It is your memory: if your context resets, re-read this file and Section 2, then resume.
10. **Problems.** Diagnose and fix failures yourself first. Append each problem to `report/notes.md` (what went wrong, why, fix). This feeds the "what went wrong" part of the report. Ask me only when I am the blocker.
11. **Report text is written as you go.** After each phase, append that phase's chapter text to `report/report.md` (format in Phase 12) so nothing is left for the end.
12. **Training limits (keep Colab time short, results unaffected).** Full runs and the two final runs: max 20 epochs, EarlyStopping patience 3, ReduceLROnPlateau patience 1. Fine-tuning stage: max 10 epochs. Tuner trials: max 5 epochs, patience 2. Smoke test: 1 epoch. Do not use mixed precision. Keep all seven runs R0 to R6 (the brief needs at least 6 runs that each differ by one change). Put these as constants in one place and mention them in the report.
13. **Repository hygiene (push only what the brief requires).** Create no file or folder that the brief or this prompt does not need. No placeholder files (`.gitkeep`), no notes, logs, temp or extra docs in the repo.
   - **Allowed in git:** `src/`, `app/`, `api/` (code, `Dockerfile`, `requirements.txt`, `README.md` Space metadata), `tests/`, `splits/`, `notebooks/colab_train.ipynb`, `samples/` (about 6 small images), `.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.gitignore`, `.env.example`, `requirements*.txt`, `pins.json`, `README.md`, and the DVC files (`data/raw.dvc`, `data/.gitignore`, `.dvc/config`, `.dvcignore`).
   - **Never in git (all git-ignored):** `.env`, `.venv/`, `data/archive.zip`, `data/raw/`, `model/`, `api/model/` and any file copied into `api/` for deployment, `mlruns/`, `scratch/`, `report/` (progress, notes, evidence, screenshots, figures, tables, report, PDF), `outputs/`, `postman/`, caches, logs, tokens.
   - Any scratch or working file goes in `scratch/` (ignored) or is **deleted before pushing**. Toy labs live in `scratch/` as separate throwaway repos.
   - Before every commit run `git status --short`; before every push run `git ls-files` and check it contains only allowed items. Remove or ignore anything else first.

---

## 2. Stop Protocols

### 2a. CHECKPOINT protocol (screenshots)

Every `[CP-x]` in this file is mandatory. At each one:

1. Finish all automated work first so the screen I capture is final and correct.
2. Print a block with: the CP id, the **exact command(s) I should run** in the terminal (only quick display commands, you do the heavy work), or the exact URL/page to open, the **file name(s)** from the Checkpoint Register (Appendix A) and the save folder `report/screenshots/`, and the list of things the screenshot **must show**.
3. **STOP.** Run nothing else. Wait for me to say "done".
4. When I say done, open each image file and verify it: all required items are visible and readable; context is visible (terminal: my username, date/time and project folder; browser: address bar and page title); **no token, password, API key or `.env` content is visible**. If anything fails, say exactly what is wrong and ask me to retake. Repeat until it passes.
5. Only after it passes: add a line to `report/evidence_log.md` (CP id, file, what it proves, report section), update `progress.md`, continue.

Never skip, merge, pre-approve or pretend a checkpoint. Never create placeholder or fake screenshots. Do not add checkpoints beyond Appendix A.

**Terminal evidence rule.** For terminal screenshots I run the display command myself, so my own prompt is in the picture. Always start the command with a context banner for my OS/shell (detect it in Phase 0):
- PowerShell: `Write-Host "$env:USERNAME | $(Get-Date) | $(Split-Path -Leaf (Get-Location))"`
- bash/zsh: `echo "$USER | $(date) | $(basename "$PWD")"`

### 2b. MANUAL STOP protocol

Every `[MS-x]` is something only I can do (logins, website clicks, Colab, Postman, decisions). Print: what to do in numbered steps, exactly where to click or what to run, and what to tell you when finished. Then STOP and wait. Do not ask me to do anything that you can do yourself.

---

## 3. Fixed Decisions (do not revisit)

- **Laptop (local):** all code writing, data audit, EDA, DVC, toy labs, Flask app, Docker, CI, evaluation, error analysis, report.
- **Google Colab (T4 GPU):** all VisionGuard model training and Keras Tuner. The Colab notebook clones the public GitHub repo, pulls data with DVC from DagsHub storage, trains, and logs everything to DagsHub MLflow. No file is moved by hand: split lists are in git, data is in DVC, models/metrics/artifacts are in MLflow. Locally you read results back from DagsHub.
- **Smoke test first:** before I use Colab, run the full training code locally on a tiny subset (`--smoke`: about 16 images per split, 1 epoch, logging to a local `mlruns/`) so Colab does not fail on silly bugs.
- **Models:** (1) baseline CNN from scratch; (2) MobileNetV2 transfer learning (frozen, then fine-tune last block). EfficientNetB0 appears only as one backbone-comparison run.
- **Label:** folder name is the label. Positive class (1) = AI-generated, 0 = real. Sigmoid output = P(AI). Reported confidence = probability of the predicted label.
- **Preprocessing lives inside the saved model** (Rescaling layer: 1/255 for the CNN; the MobileNetV2 scaling to [-1, 1], same as `mobilenet_v2.preprocess_input`; EfficientNetB0 has it built in). The model takes raw 0-255 pixels at a fixed size. One shared function (`decode + resize`, same TF ops) is used by training and by serving so they cannot drift.
- **Flask inference mode** from env `INFERENCE_MODE`: `local` (loads `model/champion.keras` once at start-up; used for Tasks 4 to 6) or `remote` (sends the image to the cloud API; Task 8). Imports of TensorFlow happen only in `local` mode so tests and CI run without TensorFlow or a model file.
- **Database:** Flask-SQLAlchemy, one `predictions` table. `DATABASE_URL` defaults to SQLite for local runs and tests, Postgres in Docker Compose.
- **Server:** Flask server, `debug` off (the brief says so).
- **Model delivery:** the champion is downloaded once from the DagsHub registry (by alias) into `model/champion.keras`. Compose **mounts** it read-only; the cloud API **bakes it in** (the Hugging Face Space builds its image from the uploaded `api/` files, which include the model). No server ever calls DagsHub per request.
- **Cloud (Task 7): Pathway C ("any other cloud"), Hugging Face Spaces (Docker Space, free CPU, no payment card)** for the model API only. Google Cloud was dropped because I cannot complete its billing step. Flask UI and Postgres stay on my localhost (the Task 7/8 text says Flask runs on localhost and calls the hosted API). In the report, map Compose elements to the Space: web inference part to the Docker Space, database stays local, why. **State openly in the report** that Hugging Face is not one of the clouds named in the brief and that this is my reading of Pathway C and the Task 7 text. Pathway C also mentions managed PostgreSQL; do not claim that the local Postgres satisfies that. If my instructor says another host or a managed database is required, stop and discuss options (for example a free managed Postgres service) before changing anything.
- **GitHub repo is public** so Colab can clone without a token.

---

## 4. Phases

Work in order. Update `progress.md` and commit/push at the end of each phase.

### Phase 0 - Kickoff

0.1 Read the brief. Create `report/progress.md`, `report/notes.md`, `report/evidence_log.md`, folders `report/screenshots`, `report/figures`, `report/tables`, `outputs`.
0.2 Detect my OS and shell. Check installed tools: `git`, `python` (3.10 to 3.12), `docker` (running), `gh`. If any are missing, tell me which ones (only those) and wait. If TensorFlow will not install natively on Windows, use WSL2 or Docker for the local smoke tests (training runs on Colab, so no local GPU is needed).
0.3 **[MS-0]** Ask me once for: full name, roll number, GitHub username, DagsHub username, Hugging Face username, instructor page limit for the report (or "none").
0.4 **[MS-1] Colab version check.** Give me this snippet to run in a blank Colab notebook and paste the output back:
```python
import sys, tensorflow as tf, keras
print(sys.version.split()[0], tf.__version__, keras.__version__)
```
Pin that TensorFlow and Keras version everywhere. Choose the local Python version that supports it.
0.5 Check `.env` exists with `DAGSHUB_USER`, `DAGSHUB_TOKEN`, `HF_USER` and `HF_TOKEN` (never print values; the Hugging Face token must have write permission). Check `data/archive.zip` exists; extract to `data/raw/`. Inspect the real folder names (do not assume them); the folder is the label.
0.6 `git init`; write `.gitignore` (`.env`, `.venv/`, `data/archive.zip`, `model/`, `api/model/`, `mlruns/`, `scratch/`, `report/`, `outputs/`, `postman/`, `.ipynb_checkpoints/`, `.pytest_cache/`, `.ruff_cache/`, `*.log`, `__pycache__/`, `*.pyc`). Do not ignore `data/raw`; DVC handles that in Phase 1. Create the GitHub repo `<ROLL>_A2_Ops` as **public** with `gh repo create ... --public --source . --push`. If `gh` is not logged in, **[MS]** ask me to run `gh auth login --scopes workflow` (the `workflow` scope is needed to push the CI workflow file).

### Phase 1 - Task 1: Data Understanding and Versioning

1.1 Create `.venv`. Write `requirements.txt` (runtime, exact pins: flask, flask-sqlalchemy, psycopg binary driver, pillow, numpy, requests, python-dotenv, `tensorflow-cpu`; use `tensorflow` locally only if the platform requires it), `requirements-train.txt` (for Colab, **no TensorFlow**: `keras` pinned, mlflow, dvc, keras-tuner, scikit-learn, pandas, matplotlib, python-dotenv), `requirements-dev.txt` (everything locally incl. pytest, ruff). Install. Write the pinned tensorflow, keras and mlflow versions to `pins.json`. Save a versions + GPU check output to `report/tables/versions.txt`. **[CP-1.1]**

1.2 `src/audit.py`: walk `data/raw`; for every image record class, file size, width, height, colour mode, format, SHA-256. Detect unreadable files (open + verify) and exact duplicates (same hash; flag cross-class duplicates separately because they are label conflicts). Save `report/tables/audit.csv` and `report/tables/audit_summary.md` with: counts per class; min, median, max of width, height, file size (per class); number of corrupt files; number of duplicates.

1.3 `src/eda.py`: a grid of sample images per class; histograms of width, height and file size per class. Save PNGs in `report/figures/`. Write 2-3 sentences of interpretation per figure (note any class difference in size or format that a model could exploit as a shortcut).

1.3b `src/make_splits.py`: drop corrupt files; keep one copy of each duplicate (drop all copies of cross-class conflicts); stratified 70/15/15 split, seed 42; write `splits/train.csv`, `val.csv`, `test.csv` with columns `relpath,label`. Print counts. Commit the splits.

1.4 **DVC.**
- *Toy part* in `scratch/dvc_toy/` (new empty folder, its own git): small CSV, `git init`, `dvc init`, `dvc add`, commit v1, edit CSV, `dvc add`, commit v2, go back to the older version (`git checkout <v1> -- <file>.dvc` + `dvc checkout`), show the content differs, return to latest. Create the DagsHub repo `<ROLL>_A2_Ops_dvc_toy` through the DagsHub API (`POST https://dagshub.com/api/v1/user/repos`, basic auth user + token). If that fails, **[MS]** ask me to create an empty repo with that exact name. Push git (without writing the token into any file or printing it; for example a one-off `http.extraHeader` Basic auth built from `.env`). Configure DVC remote = DagsHub storage (`https://dagshub.com/<user>/<repo>.dvc`, credentials only with `dvc remote modify --local`), `dvc push`, delete the local data, `dvc pull`.
  - **[CP-1.4A]** (local toy workflow, 2 images). **[CP-1.4B]** (DagsHub toy repo page).
- *Apply part:* **[MS-2]** Tell me to create the DagsHub repo `<ROLL>_A2_Ops` by connecting it to my GitHub repo (DagsHub: New Repository, Connect a repository, GitHub, pick `<ROLL>_A2_Ops`). Give exact steps. Wait. Verify it exists via API.
  Then in the main repo: `dvc init`, `dvc add data/raw`, remote `origin` = DagsHub storage (credentials with `--local`), `dvc push`, commit `data/raw.dvc`, `data/.gitignore`, `.dvc/config`, push to GitHub. In the report explain what the pointer file holds (hash, size, file count, path) and why large data does not belong in Git. **[CP-1.4C]** (2 images).

### Phase 2 - Tool Labs: MLflow (3.1, 3.2)

3.1 `src/toy_mlflow.py`: tiny Keras model on a Fashion-MNIST subset. Log params (learning rate), per-epoch metrics, one plot, the model. Run twice with different learning rates against local `mlruns/` (experiment `toy_local`). Start `mlflow ui` on port 5001. Put my own-words definitions (experiment, run, parameter, metric, artifact, tag) in the report text (simple student wording). **[CP-3.1]** (3 images).

3.2 Add to `.env` (copy values from existing ones, never print): `MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<ROLL>_A2_Ops.mlflow`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (the token). Code reads them with python-dotenv. Prove `.env` is ignored (`git check-ignore -v .env`). Repeat the toy run against DagsHub. **Also in this toy run** do `log_model`, register it, set an alias, load it by alias. This de-risks 3.6. If DagsHub refuses aliases, stop and tell me the exact error and your proposed fallback; do not silently change the requirement. **[CP-3.2]** (3 images).

### Phase 3 - Task 2: Model Code (local, smoke-tested)

Write the code now; the real training happens in Colab.

2.1 `src/data.py`: `tf.data` pipeline from the split CSVs: read, decode (3 channels, no animation), resize to 224x224 (decide what to do with aspect ratio using the EDA, and explain), keep 0-255 float, cache, shuffle (train only, seeded), batch 32, prefetch. Random augmentation (flip, small rotation, zoom, brightness/contrast) on **training only**, applied after the cache. `src/show_batch.py` saves one augmented batch to `report/figures/augmented_batch.png`. Explain every stage in the report.

2.2 `src/models.py`: `build_baseline_cnn`: Rescaling(1/255), 3-4 blocks of Conv, BatchNorm, ReLU, MaxPool, GlobalAveragePooling, Dropout, Dense(1, sigmoid). Loss binary cross-entropy, Adam, metrics accuracy, AUC, precision, recall; justify each choice and each layer. `src/show_summary.py` prints the summary and saves a per-layer table (output shape, params) to `report/tables/baseline_summary.csv`. **[CP-2.2]**

2.3 In `src/models.py`: `build_transfer(backbone, img_size, dropout)`: pretrained backbone from `keras.applications` (`include_top=False`, ImageNet weights), scaling inside the model as in Section 3, frozen backbone, call with `training=False`, head = GlobalAveragePooling, Dropout, Dense(1, sigmoid). `unfreeze_last_block(model)` unfreezes only the last block (keep BatchNorm layers frozen) and recompiles with a much smaller learning rate. A helper lists the trainable layers of each stage (save to `report/tables`). Explain why the model's expected preprocessing is used.

2.4 `src/train.py`: `train_run(config)` with fixed seeds, EarlyStopping (val_loss, restore best), ModelCheckpoint (val_loss, best only), ReduceLROnPlateau. Return history and best epoch. Plot loss and accuracy curves with the best epoch marked, save to `report/figures/`.

2.7 Toy: `src/toy_tuner.py` Keras Tuner (RandomSearch) over learning rate and units on a Fashion-MNIST subset, a few trials, save `report/tables/toy_tuner.csv`, print best settings. Run locally.

`src/experiments.py` (runs on Colab; has `--stage runs|tuner|finals|all` and `--smoke`; **idempotent**: a run is skipped only when a FINISHED run with the same name exists AND its tags `config_hash` (hash of the run's full config), `code_commit` (last commit touching `src/data.py`, `src/models.py`, `src/train.py`, `src/experiments.py`, `src/tuner.py`, `requirements-train.txt`: `git log -1 --format=%H -- <those files>`) and `dvc_data_md5` all match the current values; otherwise it trains a new run with the same name and sets tag `superseded=true` on the older ones. Comparison, registry and evaluation use only the latest non-superseded FINISHED run per name). It must **never touch the test set**.
- Stage `runs` (every run logs all hyperparameters, per-epoch train/val metrics, best epoch, tags `git_commit`, `dvc_data_md5` (read from `data/raw.dvc`), `seed`; artifacts: curves figure, split lists, `pip freeze` file, best model via `mlflow.keras.log_model`):
  - `R0_baseline_cnn`
  - `R1_tl_frozen_lr1e-3` (reference)
  - `R2_tl_frozen_lr1e-4` (learning rate only)
  - `R3_tl_frozen_noaug` (augmentation only)
  - `R4_tl_frozen_img160` (image size only)
  - `R5_tl_finetune_last_block` (frozen layers only: continue R1, unfreeze last block, small LR)
  - `R6_effb0_frozen_lr1e-3` (backbone only)
- Stage `tuner` (`src/tuner.py`): Keras Tuner RandomSearch, **exactly 8 trials** (the brief requires at least 8), searching learning rate, dropout, number of unfrozen layers; objective val_loss; short epochs with early stopping. One MLflow run `tuner` with best params and the trials table as artifact; also save the table to `report/tables`. In the report say which setting mattered most and why choosing the best validation score can overfit the validation set.
- Stage `finals`: train the best tuner configuration twice with the same seed: `final_manual` (manual Keras callback that logs params, per-epoch metrics, tags) and `final_autolog` (`mlflow.tensorflow.autolog()`; if it is incompatible with the installed Keras, use `mlflow.keras.autolog()` and note it). Set the same tags in both. Log the model in both. Explain what each approach captures or misses (autolog does not know the git/DVC tags or test metrics unless you add them).

Smoke test everything locally with `--smoke`. Then write `notebooks/colab_train.ipynb` (create with `nbformat`, no outputs) with these cells: (1) print versions; (2) clone the public repo (or `git pull` if present) and `pip install -r requirements-train.txt` (never reinstall TensorFlow), then `python src/check_versions.py`, which compares tensorflow, keras and mlflow with `pins.json` and stops with a clear message on any mismatch; (3) read `DAGSHUB_USER`/`DAGSHUB_TOKEN` from Colab Secrets (`google.colab.userdata`) into environment variables, set the MLflow variables, configure the DVC remote with `--local` credentials, `dvc pull data/raw.dvc`; (4) GPU check; (5) `python src/experiments.py --stage runs`; (6) `--stage tuner`; (7) `--stage finals`; (8) print a summary of runs and `ALL DONE`. Commit and push.

**[MS-3] Colab run.** Give me the exact Colab URL (`https://colab.research.google.com/github/<GH_USER>/<ROLL>_A2_Ops/blob/main/notebooks/colab_train.ipynb`) and steps: set GPU runtime, add the two Colab Secrets with notebook access on, Run all, keep tab open and laptop awake. Tell me to say "started". **Do not wait idly**: continue with Phase 4 immediately while it runs. If I paste an error, fix, push, and tell me to re-run from the top.

### Phase 4 - While Colab trains (nothing here needs the champion model)

**4a. Task 6.1 Docker warm-up** with a ready-made image (for example `nginx`): run a container, map a port, mount a volume, read logs, open a shell inside, list and remove containers and images. Write my-own-words definitions (image, container, layer, volume, port mapping) in the report. Prepare each step so one screen shows the action. **[CP-6.1A]** run + port + volume. **[CP-6.1B]** logs + shell. **[CP-6.1C]** list + remove.

**4b. Task 6.6 toy CI** in `scratch/ci_toy/`: create throwaway repo `<ROLL>_A2_Ops_ci_toy` with `gh`. Workflow on every push, matrix of two Python versions, runs a small test file. First push a failing test. **[CP-6.6A]** red run. Fix, push. **[CP-6.6B]** green run.

**4c. Build the application code** (no screenshots yet, unit tests with a fake predictor):
- `app/` Flask app factory `create_app()`, `config.py` (all settings from env: `MODEL_PATH`, `THRESHOLD`, `DATABASE_URL`, `MAX_UPLOAD_MB`, `INFERENCE_MODE`, `API_URL`, `API_KEY`), `inference.py`, `validators.py`, `models.py`, `routes.py`, `templates/` (`base.html`, `upload.html`, `result.html`, `history.html`, `error.html`), `static/style.css`.
  - **4.1** `inference.py`: loads the model **once** at start-up, preprocesses uploaded bytes with the shared function, returns label, confidence, p_ai, model name and version. Report: why per-request loading is unacceptable and why training and serving preprocessing must match. A small `python -m app.inference <image>` prints the result for demos.
  - **4.2** Routes: upload page, result, history, health (`/health` returns JSON with status, model loaded, database ok). Reject wrong file types, files over the size limit (`MAX_CONTENT_LENGTH`) and files that look like images but cannot be decoded (decode in memory). Sanitise filenames (`secure_filename`). **Never write the upload to disk.** Friendly error pages for 400, 413, 415, 500.
  - **4.3** Simple HTML/CSS: nav (Upload, History), upload form, result page (verdict, confidence, P(AI), an honest caveat that detectors can be wrong), history table (time, filename, verdict, confidence, model name + version, inference mode).
  - **4.4** Run with the Flask server, `debug` off. Report: why debug is off (the interactive debugger allows code execution, leaks stack traces, reloader overhead).
  - **Task 5** `models.py`: `Prediction` table (id, UTC timestamp, sanitised filename, label, confidence, p_ai, model name, model version, inference mode). Insert one row after every successful prediction (not for rejected uploads). History shows newest first. Explain the table design in the report.
  - `remote` mode: `requests.post(API_URL + "/predict", files=..., headers={"X-API-Key": ...}, timeout=90)`; friendly page on timeout, 401, 5xx; save model fields from the API response.
- `api/` (Task 7): `app.py` with `GET /health` and `POST /predict` (multipart `file`, header `X-API-Key` compared in constant time to env `API_KEY`; same validation as the web app; same preprocessing; JSON response with label, confidence, p_ai, threshold, model name and version); log each request (time, label, confidence, never the image) to stdout. `api/Dockerfile` (slim base, `tensorflow-cpu`, a non-root user with **UID 1000** (required by Spaces), model baked into `/app/model/champion.keras`, listens on **port 7860**, debug off), `api/requirements.txt` (minimal), `api/README.md` containing only the Space metadata (`sdk: docker`, `app_port: 7860`, a title). `src/deploy_api.py` copies `model/champion.keras` and the shared preprocessing file into `api/` (both git-ignored) and deploys to the Space with `huggingface_hub` (Phase 9).
- `Dockerfile`, `.dockerignore`, `docker-compose.yml` (see Phase 7), `.github/workflows/ci.yml` (see Phase 8), `tests/` (validators, routes with a fake predictor, health, history insert), `postman/VisionGuard.postman_collection.json` and a Postman environment template with empty `base_url` and `api_key` variables. Requests: Health; Predict real image; Predict AI image; Predict without key (expect 401); Predict wrong file type (expect 400 or 415).
- `README.md`: what it is, how to run locally, with Compose, tests, links; CI badge added in Phase 8.

### Phase 5 - After Colab (I say "Colab done")  **[MS-3b]**

5.1 Query DagsHub MLflow through the Python client. Check that all runs (R0 to R6, tuner, final_manual, final_autolog) are FINISHED and have params, per-epoch metrics, tags and artifacts. If anything is missing, fix the code, push, and tell me which Colab cells to re-run. Download curves and figures from MLflow into `report/figures/`.

5.2 **3.5 Comparison.** Build one table and one chart of R0 to R6 (and the finals) with the key validation metrics: `report/tables/run_comparison.csv` and a figure. Choose the best pre-tuning run among R1 to R6 using **validation** metrics (primary: val ROC-AUC and val F1; secondary: val loss) and justify. **[CP-3.5]**

5.3 **3.3** Explain manual vs autolog logging in the report using the real run pages. **[CP-3.3]** (2 images).

5.4 **3.6 Registry.** `src/register.py`: register one model `<ROLL>_A2_Ops_visionguard` with at least two versions: v1 = best of R1 to R6, v2 = `final_manual`. Give the better one (by validation metrics) the alias `champion`. **[CP-3.6A]**

5.5 `src/download_champion.py` saves the champion into `model/champion.keras`. Separate small script `src/predict_by_alias.py <image>` loads the model **by alias** (`champion`) and classifies one image from `samples/`. **[CP-3.6B]**

5.6 **2.5 Evaluation** `src/evaluate.py`:
- Choose a separate decision threshold for **each model** (baseline CNN and champion) on that model's own **validation** predictions (state the rule, make it consistent with the worse-error decision below).
- Evaluate on the **test set exactly once per model**: baseline CNN (best checkpoint of R0) and the champion, each with its own chosen threshold. Metrics at that threshold: accuracy, precision, recall, F1, ROC-AUC, plus confusion matrix and ROC curve figures. Table for both models in `report/tables/test_metrics.csv`.
- Guard against reuse: compute all test predictions and metrics for both models first, save figures and tables, log to MLflow, and only when everything has succeeded write `outputs/test_done.json` as the very last step. If that file exists, refuse to run test evaluation again unless I explicitly say so. If a technical failure happens before the file is written, fix the bug and rerun (note it in `notes.md`); never change any choice because of test results.
- Log test metrics to the matching MLflow runs (champion run = `final_manual` or the chosen run, baseline run = R0). **3.4 Artifacts:** attach to the champion run: confusion matrix, ROC curve, split lists, package list (`pip freeze`), the Keras model.
- Write the justification: positive class = AI-generated; decide which error is worse for this product (calling a real photo fake, or a fake photo real) and justify. I will review it.
- Put the champion model's threshold in `.env` as `THRESHOLD` (example value in `.env.example`). The Flask app, Compose and the Space API all use this same value.
5.7 **2.6 Error analysis** `src/errors.py`: a grid of 10 misclassified test images (all of them if fewer than 10) with true label, predicted label, probability; look at each image and write a short hypothesis for each (blur, painting style, watermark, size, format, from the audit table).
5.8 Copy about 6 test images (3 real, 3 AI) into `samples/` for demos (commit). **[CP-3.4]**

### Phase 6 - Flask app checks (Tasks 4 and 5)

Run locally: `INFERENCE_MODE=local`, SQLite, debug off, port 5000.
- **[CP-4.1]** two known images (one real, one AI) through the inference module in the console.
- Create bad test files in `scratch/bad_samples/` (a `.txt` file, an oversize image above the limit, a text file named `fake.jpg`) with a small script. **[CP-4.2]** (3 images).
- Run the unit tests. **[CP-4.3]** upload page, result page, history page with at least 3 rows, `/health` (4 images; this also covers 4.4).

### Phase 7 - Docker (Tasks 6.2 to 6.5)

6.2 `Dockerfile`: slim Python base matching the pinned Python; copy `requirements.txt` and install **before** copying source (layer cache); `tensorflow-cpu` (explain the choice); non-root user `appuser`; `HEALTHCHECK` (python-based, since slim has no curl); `.dockerignore` (`.git`, `.venv`, `data`, `mlruns`, `scratch`, `report`, `notebooks`, `model`, `.env`, `__pycache__`); no secrets in the image. **6.4 before:** build first **without** `--no-cache-dir`; record image size and the largest layers (`docker images`, `docker history`). **After:** add `--no-cache-dir`, rebuild, record size. Two numbers and a conclusion in the report. **[CP-6.2]**

6.3 `docker-compose.yml`: top-level `name:` set to `<SLUG>` (lowercase); services `web` and `db` (`postgres` alpine). `db`: named volume, `pg_isready` healthcheck, **no `ports`**, credentials via `${POSTGRES_USER}`, `${POSTGRES_PASSWORD}`, `${POSTGRES_DB}` (Compose reads these from the untracked `.env` for substitution). `web`: `depends_on` db with `condition: service_healthy`; **no `env_file`**; an explicit `environment:` list with only `DATABASE_URL` (built from the `${POSTGRES_*}` values, host `db`), `THRESHOLD`, `MODEL_PATH`, `MAX_UPLOAD_MB`, `INFERENCE_MODE`, `API_URL`, `API_KEY`; publishes 5000, mounts `./model:/app/model:ro`. `DAGSHUB_*` and `MLFLOW_*` must never reach any container. Justify mounted vs built in. Provide `.env.example`. Start it. Prove no DagsHub or MLflow variable reached the container with `docker compose exec web sh -c 'env | cut -d= -f1 | sort'` (names only, never values) and describe it in the report. **[CP-6.3]** (2 images).

6.5 Demonstrate start, stop, restart, logs and a shell in each service (capture text output for the report, no screenshot). Show a Postgres query of the `predictions` table and that data survives `down` and `up` (text for Task 5). Then break something on purpose (wrong database password for the web service, or missing model file), diagnose **only from `docker compose logs`**, fix, and record log excerpt, diagnosis and fix. Restore a healthy stack.

### Phase 8 - CI for the project (6.6 apply)

`.github/workflows/ci.yml`: on push and pull request; checkout, setup Python, install `requirements.txt` plus `pytest` and `ruff`, run `ruff check`, run `pytest`, build the Docker image. Add the status badge to `README.md`. Push, watch with `gh run watch`, fix until green. In the report state: workflow secrets live in the repository's Settings, Secrets and variables, Actions (encrypted, masked in logs); they must not be in the workflow file because the file is versioned and readable; this workflow needs none. **[CP-6.6C]** (2 images).

### Phase 9 - Task 7: Cloud API (Hugging Face Space)

No manual step is needed here (the account and token already exist in `.env`). Do everything with the `huggingface_hub` Python API in `src/deploy_api.py` (add `huggingface_hub` to `requirements-dev.txt` only):

1. Generate the API key with `secrets.token_urlsafe` into `.env` (`API_KEY`, never printed). **Never put the key in a command line, a file inside `api/`, or output.**
2. `create_repo(<hf_user>/<SLUG>-visionguard-api, repo_type="space", space_sdk="docker", exist_ok=True)` as a **public** Space (the API is public but every predict call needs the key).
3. Set `API_KEY` with `add_space_secret` (value read from `.env` in Python, never printed) and `THRESHOLD` with `add_space_variable`.
4. Copy the model and shared preprocessing into `api/`, then `upload_folder` of `api/` only (never `.env`, data or the repo root). Large files are handled by the library.
5. Poll `get_space_runtime` until the stage is `RUNNING` (the build installs TensorFlow and can take several minutes). If the build fails, read the Space build logs, fix, upload again.
6. Save the public URL to `.env` as `API_URL`. Test with `curl` using a real and an AI sample, and a call without key (expect 401). The first request after the Space sleeps can take about a minute; retry before concluding it is broken.

**[CP-7.1]** (1 image: the Space page with name, Docker SDK and status Running). **[CP-7.2]** (2 images: terminal output of a small script printing the Space URL and runtime stage `RUNNING`; the Space Settings page showing `API_KEY` listed as a secret with the value hidden).

**[MS-5] Postman.** I import the collection and environment, put the URL and key in variables (key variable marked as secret), and run all requests **from another network** (phone hotspot). **[CP-7.3]** (5 images).

Take the container logs from the Space (via `huggingface_hub` or the Space Logs tab), put an excerpt in the report as "logged result". **[CP-7.4]**

Report for Task 7: API URL, step-by-step deployment with the screenshots, the Compose-to-Space mapping, how the champion reaches the server (baked in, no DagsHub call per request), the free-tier facts (free CPU Basic Space with 2 vCPU and 16 GB RAM, no payment card, sleeps after roughly 48 hours without requests and wakes on the next one; check the current Hugging Face docs and state what you found), the Pathway C interpretation (see Fixed Decisions), why Google Cloud was not used, and that the API key lives in a Space secret.

### Phase 10 - Task 8: Flask + Cloud API

Set `.env`: `INFERENCE_MODE=remote`, `API_URL`, `API_KEY`. Run `docker compose up -d --build` so Flask runs on localhost with Postgres history. **[MS-6]** I upload at least 5 different images from `samples/` (mix of real and AI). **[CP-8]** (3 result pages with address bar visible, then the history page showing all rows with decision, time, model, confidence). Report: how the Flask view calls the API, the local/remote switch, error handling.

### Phase 11 - Optional bonus

**[MS-7]** Ask me: "Do you want the Grad-CAM bonus (up to +1)?" Only if yes: `src/gradcam.py` with 4 correct and 4 wrong test predictions, one figure, a comment on what the model looks at. No screenshots needed.

### Phase 12 - Report and Final Verification

Report source `report/report.md`, final PDF `report/<ROLL>_VisionGuard.pdf` (build with the simplest reliable tool you can run here, for example Markdown to HTML to PDF). Respect the page limit I gave.
- **Front matter:** name, roll number, links (GitHub, DagsHub, live API URL), commit hash of the graded version, disclosure of AI assistance (Antigravity with a Gemini model for code generation, debugging and first drafts; ChatGPT and Claude for planning and reviewing this execution prompt and the manual guide; I reviewed and ran everything). Ask me to confirm the list and wording against what I actually used.
- **One chapter per task**; under each sub-task ID: what was done, why, how it works in simple words, what went wrong (from `notes.md`). Explain every command, setting and design choice. No unexplained pasted code. Short snippets only with commentary.
- **Screenshots:** from `report/screenshots/`, every one captioned ("Figure n: ..."), referenced in the text, and only those required by the brief.
- Include all tables and figures the brief lists as evidence (1.2 table, 1.3 figures, 2.1 batch figure, 2.2 table, 2.3 curves and trainable layers, 2.4 curves with best epoch, 2.5 metrics and written justification, 2.6 grid, 2.7 tables, 6.4 size numbers, 6.5 log excerpt, 7 Postman results).

**[MS-8]** Show me the draft and ask me to review it; the explanations must be in my own words, so apply my edits.

Final verification (write `report/final_check.md`): go through the brief row by row (every ID in every task table, Tasks 5, 7, 8, Section 11) and mark evidence present or missing. Check: all 28 checkpoints logged in `evidence_log.md`; `git status` clean and `git ls-files` contains only the Rule 13 allowed items; no secret in the repo or its history (search for the actual secret values without printing them); the API key exists only in `.env` and as a Space secret (not in any uploaded file or plain Space variable); the `web` container env-name list has no `DAGSHUB_*` or `MLFLOW_*`; `.env.example` complete; README has badge and run steps; CI green; `docker compose up` works from a clean clone with `.env.example` copied; DagsHub project readable (ask me to set it public or shared); API URL answered from another network (CP-7.3). Tag the final **code** commit `graded`, record its hash in the report front matter. The PDF is not committed (`report/` is git-ignored); I submit it separately. Finish with a summary: GitHub link, DagsHub link, API URL, commit hash, PDF path, and anything I must still submit.

---

## Appendix A - Checkpoint Register (28 checkpoints, 50 images)

File name pattern: `CP-<id>_<letter>_<keyword>.png`, saved in `report/screenshots/`. Always give me the exact name.

| CP | Images and what they must show |
|---|---|
| CP-1.1 | a_versions: terminal with Python version, versions of tensorflow, keras, numpy, pandas, pillow, scikit-learn, mlflow, dvc, flask, and the GPU check result |
| CP-1.4A | a_toy_commits: `git log --oneline` with 2 commits and the older CSV version restored, then latest again. b_toy_pushpull: `dvc push`, local data deleted, `dvc pull` restoring it |
| CP-1.4B | a_toy_dagshub: DagsHub toy repo page with commits and the `.dvc` file, address bar visible |
| CP-1.4C | a_pointer: terminal with contents of `data/raw.dvc` and `dvc status -c` up to date. b_dagshub_data: DagsHub main repo Files tab showing `data/raw` tracked by DVC |
| CP-2.2 | a_summary: full baseline CNN `model.summary()` incl. total parameters |
| CP-3.1 | a_run1 and b_run2: the two local MLflow run pages (params, metrics, artifacts). c_compare: comparison view of both runs |
| CP-3.2 | a_dagshub_project: DagsHub project home with repo name and GitHub connection. b_toy_run: toy run in DagsHub Experiments. c_env_ignored: terminal with `git check-ignore -v .env` and `git status --short` not listing `.env` |
| CP-3.5 | a_compare: DagsHub experiments table or compare view with at least 6 runs and metric columns |
| CP-3.3 | a_manual_run: `final_manual` run page with params, metric curves, tags. b_autolog_run: `final_autolog` run page with params, metric curves, tags |
| CP-3.6A | a_registry: DagsHub model page for `<ROLL>_A2_Ops_visionguard` with 2+ versions and alias `champion` |
| CP-3.6B | a_predict_by_alias: terminal output of loading by alias and classifying one image |
| CP-3.4 | a_artifacts: champion run Artifacts tab with confusion matrix, ROC, split lists, package list, model |
| CP-4.1 | a_two_images: terminal output of the inference module for two known images (label + confidence) |
| CP-4.2 | a_wrong_type, b_too_big, c_corrupt: friendly error pages in the browser for the three rejected uploads, address bar visible |
| CP-4.3 | a_upload: upload page. b_result: result page. c_history: history with 3+ rows (decision, time, model, confidence). d_health: `/health` JSON, address bar visible |
| CP-6.1A | a_run_port_volume: container run with `-p` and `-v`, `docker ps` showing the mapping |
| CP-6.1B | a_logs_shell: `docker logs` and a shell prompt inside the container |
| CP-6.1C | a_list_remove: `docker ps -a`, `docker images`, remove container and image, listing afterwards |
| CP-6.6A | a_toy_red: GitHub Actions toy repo, red run, matrix of two Python versions |
| CP-6.6B | a_toy_green: same repo, green run |
| CP-6.2 | a_nonroot: `whoami` / `id` inside the container showing a non-root user |
| CP-6.3 | a_compose_ps: `docker compose ps` with both services healthy and no host port for `db`. b_db_closed: connection attempt to `localhost:5432` from the host failing |
| CP-6.6C | a_project_green: green Actions run of the VisionGuard repo. b_badge: GitHub README showing the badge |
| CP-7.1 | a_space_running: Hugging Face Space page showing the Space name, Docker SDK and status Running, address bar visible |
| CP-7.2 | a_deploy_terminal: terminal with the Space URL and runtime stage RUNNING from a small script. b_space_settings: Space Settings, Variables and secrets, showing `API_KEY` as a secret with the value hidden |
| CP-7.3 | a_health: Postman GET /health 200. b_predict_real and c_predict_ai: Postman predict responses. d_no_key_401: request without key. e_other_network: proof of another network (phone hotspot name visible plus a public IP page) |
| CP-7.4 | a_cloud_logs: Hugging Face Space Logs (container) showing the Postman requests |
| CP-8 | a_result1, b_result2, c_result3: result pages for three different images (at least one real, one AI), address bar visible. d_history: history page with all 5+ rows |

Execution order of checkpoints: 1.1, 1.4A, 1.4B, 1.4C, 3.1, 3.2, 2.2, then (Colab running) 6.1A, 6.1B, 6.1C, 6.6A, 6.6B, then (Colab done) 3.5, 3.3, 3.6A, 3.6B, 3.4, 4.1, 4.2, 4.3, 6.2, 6.3, 6.6C, 7.1, 7.2, 7.3, 7.4, 8.
