# Models

The completed modelling pipeline writes fitted Linear Regression and XGBoost artifacts under `models/one_step/`. These generated binaries remain ignored because the public dashboard reads validated saved predictions and metrics, not fitted estimators.

Running `src/train_one_step_models.py` during full pipeline reproduction recreates the artifacts needed by `src/run_recursive_forecasts.py`.
