# VisionGuard Engineering Notes and Problem Log

This log documents issues, root causes, and fixes encountered during development, serving as the record for the "what went wrong" sections of the final report.

| Incident # | Phase / Task | Issue Description | Root Cause | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Phase 1 / Task 1.1 | Dependency conflict between `tensorflow-cpu==2.21.0` and `mlflow==2.20.2`. | `tensorflow-cpu 2.21.0` requires `protobuf>=6.31.1`, but `mlflow 2.20.2` strictly required `protobuf<6`. | Upgraded pinned MLflow version to `3.18.0` in `requirements-train.txt` and `pins.json`, which natively supports Protobuf 6.x. |
| 2 | Phase 1 / Task 1.4 | DVC crashed with `cannot import name '_DIR_MARK' from 'pathspec.patterns.gitwildmatch'`. | `pathspec` 1.1.1 removed internal attribute `_DIR_MARK` relied upon by DVC 3.59.0. | Pinned `pathspec==0.11.2` in `requirements-train.txt` and `requirements-dev.txt`. |
| 3 | Phase 2 / Task 3.1 | MLflow raised `MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode`. | MLflow 3.18.0 deprecates legacy local file storage unless an explicit opt-in environment variable is set. | Set `MLFLOW_ALLOW_FILE_STORE=true` in environment and `src/toy_mlflow.py` to allow local `./mlruns` tracking. |
| 4 | Phase 3 / Task 2.7 | Keras Tuner raised `ModuleNotFoundError: No module named 'tensorboard'` during trial execution. | `keras-tuner` expects `tensorboard.plugins.hparams` to be available. | Installed and pinned `tensorboard==2.21.0` in `requirements-train.txt` matching TensorFlow 2.21.0. |
| 5 | Phase 3 / Task 2.4 | `TypeError: float() argument must be a string or a real number, not 'Functional'` in `experiments.py`. | `train_run` returned `(history, best_epoch, model)` instead of `best_val_loss`. | Updated `train_run` in `src/train.py` to return `float(val_losses[best_epoch - 1])` as the third return value. |
