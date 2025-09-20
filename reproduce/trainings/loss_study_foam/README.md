# Loss Weighting Study (Foam)

This folder contains scripts and notebooks for studying the effect of different loss weighting strategies in PINN training, with a focus on OpenFOAM data.

## Contents
- `adapting_inverse_lossweights.ipynb`: Study adaptive inverse loss weighting strategies.
- `adapting_smoothened_inverse_lossweights.ipynb`: Explore smoothened adaptive inverse loss weighting.
- `grid_search_cyl_loss_combinations.ipynb`: Grid search over different cylinder loss weight combinations.
- `grid_search_indv_loss_combinations.ipynb`: Grid search over individual loss weight combinations.
- `grid_search_wghts.ipynb`: General grid search for loss weights.
- `plain_lossweights.ipynb`: Baseline experiments with plain loss weights.
- `twophase_log_normalization_lossweights.ipynb`: Two-phase log normalization for loss weights.
- `wake_metrics.ipynb`: Analyze wake region metrics from experiments.

## Optimizer(s)
- Multiple optimizers for different loss weighting strategies (see optimizer files).

## How to Use
1. Open the relevant notebook for the loss weighting strategy you want to study.
2. Configure experiment parameters as needed.
3. Run the notebook to train and evaluate the model.

## Notes
- Results and metrics are saved in `losswght_trails/`.
- Useful for understanding the impact of loss weighting on PINN performance.

## Reference
- The origin repository and base code for all PINN trainings in this folder are from https://github.com/AdrianDario10/Navier_Stokes_cylinder2D