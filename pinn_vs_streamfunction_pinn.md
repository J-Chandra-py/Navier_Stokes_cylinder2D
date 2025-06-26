# Comparison: Standard PINN vs. Streamfunction PINN (PMPG) for 2D Steady Navier-Stokes

## 1. Network Output

- **Standard PINN:**  
  Neural network outputs velocity and pressure directly:  
  \[
  \text{NN}(x, y) \rightarrow (u, v, p)
  \]

- **Streamfunction PINN (PMPG):**  
  Neural network outputs only the streamfunction:  
  \[
  \text{NN}(x, y) \rightarrow \psi
  \]
  Velocities are obtained by automatic differentiation:  
  \[
  u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}
  \]

---

## 2. Enforcing Incompressibility

- **Standard PINN:**  
  Incompressibility (\(\nabla \cdot \mathbf{u} = 0\)) is enforced via a penalty term in the loss.

- **Streamfunction PINN:**  
  Incompressibility is satisfied by construction due to the streamfunction formulation.

---

## 3. Loss Function Structure

### Standard PINN Loss

\[
\mathcal{L}_{\text{PINN}} = \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}
\]

Where:
- **PDE residual (momentum):**
  \[
  \mathcal{L}_{\text{PDE}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \mathbf{u} \cdot \nabla \mathbf{u} + \frac{1}{\rho} \nabla p - \nu \nabla^2 \mathbf{u} \right|^2
  \]
- **Boundary condition penalty:**
  \[
  \mathcal{L}_{\text{BC}} = \frac{1}{N_{bc}} \sum_{i=1}^{N_{bc}} \left| \mathbf{u}_{\text{pred}} - \mathbf{u}_{\text{target}} \right|^2
  \]
- **Divergence penalty:**
  \[
  \mathcal{L}_{\text{div}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \nabla \cdot \mathbf{u} \right|^2
  \]

---

### Streamfunction PINN (PMPG) Loss

\[
\mathcal{L}_{\text{PMPG}} = \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{Eqm}} \mathcal{L}_{\text{Eqm}} + \lambda_{\text{PMPG}} \mathcal{L}_{\text{PMPG}}
\]

Where:
- **Boundary condition penalty (on u, v from ψ):**
  \[
  \mathcal{L}_{\text{BC}} = \frac{1}{N_{bc}} \sum_{i=1}^{N_{bc}} \left| \mathbf{u}_{\text{pred}}(\psi) - \mathbf{u}_{\text{target}} \right|^2
  \]
- **Equilibrium (vorticity transport) penalty:**
  \[
  \mathcal{L}_{\text{Eqm}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \nabla \times \left( (\mathbf{u} \cdot \nabla)\mathbf{u} - \nu \nabla^2 \mathbf{u} \right) \right|^2
  \]
- **PMPG (pressure gradient surrogate) penalty:**
  \[
  \mathcal{L}_{\text{PMPG}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| (\mathbf{u} \cdot \nabla)\mathbf{u} - \nu \nabla^2 \mathbf{u} \right|^2
  \]

---

## 4. Boundary Conditions

- **Standard PINN:**  
  Imposed directly on (u, v, p) outputs.

- **Streamfunction PINN:**  
  Imposed on velocities derived from ψ (and optionally on ψ itself for Dirichlet BCs).

---

## 5. Summary Table

| Aspect                | Standard PINN (u, v, p) | Streamfunction PINN (PMPG) |
|-----------------------|-------------------------|----------------------------|
| Network Output        | (u, v, p)               | ψ                          |
| Incompressibility     | Penalty in loss         | Automatic (by construction)|
| Main Loss Terms       | PDE, BC, Div            | PMPG, Eqm, BC              |
| Pressure              | Direct output           | Not explicit               |
| Boundary Conditions   | On (u, v, p)            | On (u, v) from ψ           |
| Robustness            | Sensitive to weights    | More robust for incompressible flows |

---

## 6. References

- [Your reference article or paper]
- [PINN original paper: Raissi et al., 2019]
