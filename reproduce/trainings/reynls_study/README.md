# Reynolds Number Generalization Study

This folder contains scripts and notebooks for investigating PINN generalization capabilities across different Reynolds numbers.

## Contents
- `generalisation_across_Re.ipynb`: Notebook for training and evaluating PINNs across multiple Reynolds numbers.

## Optimizer(s)
- `optimizer.py`: Standard optimizer for PINN training used across all Reynolds number experiments.

## How to Use
1. Open `generalisation_across_Re.ipynb` in Jupyter.
2. Set the desired Reynolds number range to explore.
3. Run the notebook to perform sequential training across different Re values.

## Study Objective
This study examines PINNs generalisation acorss various Reynolds numbers.

## Notes
- Results, model checkpoints and loss histories are saved in `model_data/`.
- Analysis focuses on generalization performance metrics between different flow regimes.

## Reference
- The origin repository and base code for all PINN trainings in this folder are from https://github.com/AdrianDario10/Navier_Stokes_cylinder2D