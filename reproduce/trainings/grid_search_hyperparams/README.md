# Hyperparameter Grid Search

This folder contains scripts and notebooks for systematic grid search over hyperparameters in PINN training.

## Contents
- `grid_search_paras.ipynb`: Configure and launch grid search over hyperparameters for PINN models.

## Optimizer(s)
- `optimizer.py`: Standard optimizer for PINN training, used in all grid search runs.

## How to Use
1. Open `grid_search_paras.ipynb` in Jupyter.
2. Set the hyperparameter ranges to explore.
3. Run the notebook to launch grid search experiments.

## Notes
- Results and loss trails are saved in `losstrails/`.
- Useful for identifying optimal hyperparameters for PINN models.