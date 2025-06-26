1. PDE (Interior) Loss
Purpose:
Enforce that the neural network solution satisfies the Navier-Stokes equations at interior points.

Mathematical Form:

Let

( (x_i, y_i) ) be an interior point,
( \mathbf{u}\theta(x_i, y_i) = (u\theta, v_\theta, p_\theta) ) be the network output at that point,
( \mathcal{N}(\mathbf{u}_\theta) ) be the Navier-Stokes operator (residual).
Then, for all interior points: [ \text{Loss}{\text{PDE}} = \frac{1}{N{\text{int}}} \sum_{i=1}^{N_{\text{int}}} \left| \mathcal{N}(\mathbf{u}_\theta)(x_i, y_i) \right|^2 ] where ( | \cdot | ) is typically the ( L_2 ) norm.

2. Boundary Losses
a. Inlet (Dirichlet BC)
Purpose:
Enforce prescribed velocity at the inlet (e.g., parabolic profile).

Mathematical Form: [ \text{Loss}{\text{inlet}} = \frac{1}{N{\text{in}}} \sum_{i=1}^{N_{\text{in}}} \left| u_\theta(0, y_i) - u_{\text{inlet}}(y_i) \right|^2 ] where ( u_{\text{inlet}}(y) ) is the prescribed profile (e.g., ( 4y(1-y) )).

b. Outlet (Neumann or Dirichlet BC)
Purpose:
Often, set pressure or zero-gradient at the outlet.

Mathematical Form (example, zero pressure): [ \text{Loss}{\text{outlet}} = \frac{1}{N{\text{out}}} \sum_{i=1}^{N_{\text{out}}} \left| p_\theta(x_{\text{out}}, y_i) - 0 \right|^2 ]

c. Wall (No-slip BC)
Purpose:
Enforce zero velocity at the top/bottom walls.

Mathematical Form: [ \text{Loss}{\text{wall}} = \frac{1}{N{\text{wall}}} \sum_{i=1}^{N_{\text{wall}}} \left( \left| u_\theta(x_i, y_{\text{wall}}) \right|^2 + \left| v_\theta(x_i, y_{\text{wall}}) \right|^2 \right) ]

d. Cylinder (No-slip BC)
Purpose:
Enforce zero velocity on the cylinder surface.

Mathematical Form: [ \text{Loss}{\text{cylinder}} = \frac{1}{N{\text{cyl}}} \sum_{i=1}^{N_{\text{cyl}}} \left( \left| u_\theta(x_i, y_i) \right|^2 + \left| v_\theta(x_i, y_i) \right|^2 \right) ]

3. Total Loss
All these losses are typically summed (possibly with weights):

[ \text{Total Loss} = \lambda_{\text{PDE}} \cdot \text{Loss}_{\text{PDE}}

\lambda_{\text{inlet}} \cdot \text{Loss}_{\text{inlet}}
\lambda_{\text{outlet}} \cdot \text{Loss}_{\text{outlet}}
\lambda_{\text{wall}} \cdot \text{Loss}_{\text{wall}}
\lambda_{\text{cylinder}} \cdot \text{Loss}_{\text{cylinder}} ]
where ( \lambda ) are weights (often set to 1).

How This Relates to Your Code
Each region in x_train/y_train corresponds to one of these loss terms.
For each region, the model predicts outputs, and the loss is computed as the difference from the target (zero for PDE and most boundaries, prescribed for inlet).
The optimizer minimizes the sum of all these losses.