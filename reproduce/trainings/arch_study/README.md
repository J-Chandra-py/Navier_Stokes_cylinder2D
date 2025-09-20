# Architecture Study

This folder contains scripts and notebooks for exploring different neural network architectures for PINN-based Navier-Stokes simulations.

## Contents
- `train_different_arch_combinations.ipynb`: Run and compare different neural network architectures for PINN training.

## Optimizer(s)
- `optimizer_sep_losses.py`: Allows separate loss terms for different physics constraints.
- `optimizer.py`: Standard optimizer for PINN training.

## How to Use
1. Open `train_different_arch_combinations.ipynb` in Jupyter.
2. Configure the architecture and training parameters as needed.
3. Run the notebook to train and evaluate different architectures.

## Notes
- Results and models are saved in `model_data/`.
- Useful for comparing the effect of architecture choices on PINN performance.