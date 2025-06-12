# In-Depth Explanation of the PINN Architecture for 2D Navier-Stokes Cylinder Flow

This document provides a comprehensive, detailed explanation of the Physics-Informed Neural Network (PINN) architecture and training process used in your codebase for solving the steady-state Navier-Stokes equations around a cylinder.

---

## 1. Problem Statement and Mathematical Background

### 1.1. Physical Problem

- **Objective:** Predict the steady-state velocity field (u, v) and pressure (p) for incompressible flow around a cylinder in a 2D rectangular domain.
- **Governing Equations:** The steady-state incompressible Navier-Stokes equations:

```math
\begin{align*}
&\text{Momentum equations:} \\
&\quad u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{1}{\rho} \frac{\partial p}{\partial x} - \frac{\mu}{\rho} \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0 \\
&\quad u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{1}{\rho} \frac{\partial p}{\partial y} - \frac{\mu}{\rho} \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0 \\
&\text{Continuity equation (incompressibility):} \\
&\quad \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
\end{align*}
```

- **Boundary Conditions:** 
  - **Inlet:** Prescribed velocity profile (e.g., parabolic).
  - **Outlet:** Often zero pressure or Neumann condition.
  - **Walls (top/bottom):** No-slip (u = v = 0).
  - **Cylinder:** No-slip (u = v = 0).

---

## 2. Neural Network Architecture (`network.py`)

### 2.1. Structure

- **Input Layer:** 2 neurons (x, y) — spatial coordinates.
- **Hidden Layers:** 4 fully connected (Dense) layers, each with 48 neurons.
- **Activation Functions:** Default is `tanh`, but custom `swish` and `mish` are also available.
- **Output Layer:** 3 neurons — representing u (x-velocity), v (y-velocity), and p (pressure).

#### Why this structure?

- **Dense layers:** Universal function approximators, capable of representing the complex, nonlinear solution fields of PDEs.
- **Depth and width:** 4 layers with 48 neurons each is a practical balance: deep enough to capture complex flow features (like vortices and boundary layers), but not so deep as to make training unstable or slow.
- **Activation functions:** Nonlinear activations like `tanh`, `swish`, and `mish` are crucial for learning nonlinear mappings. `tanh` is smooth and bounded, which helps with stability and gradient flow in PINNs. `swish` and `mish` can sometimes improve convergence and accuracy due to their non-monotonicity and smoothness.

#### Custom Activations

- **Swish:** 
```math
\text{swish}(x) = x \cdot \sigma(x)
```
where \( \sigma(x) \) is the sigmoid function.

- **Mish:** 
```math
\text{mish}(x) = x \cdot \tanh(\text{softplus}(x))
```
where \( \text{softplus}(x) = \log(1 + e^x) \).

#### Output Layer

- **Why 3 outputs?** The network directly predicts the three physical quantities of interest at each spatial location: u, v, and p. This direct mapping allows for efficient evaluation and differentiation.

---

## 2.1. Detailed Neural Network Architecture Graph

The PINN used for the 2D Navier-Stokes problem is a fully connected feedforward neural network (multilayer perceptron) with the following structure:

### Layer-by-Layer Description

| Layer (Type)         | Output Shape | Activation | Purpose/Notes                |
|----------------------|--------------|------------|------------------------------|
| Input (Dense)        | (None, 2)    | -          | Receives (x, y) coordinates  |
| Hidden 1 (Dense)     | (None, 48)   | tanh/swish/mish | Nonlinear feature extraction |
| Hidden 2 (Dense)     | (None, 48)   | tanh/swish/mish | Nonlinear feature extraction |
| Hidden 3 (Dense)     | (None, 48)   | tanh/swish/mish | Nonlinear feature extraction |
| Hidden 4 (Dense)     | (None, 48)   | tanh/swish/mish | Nonlinear feature extraction |
| Output (Dense)       | (None, 3)    | linear     | Predicts (u, v, p)           |

- **Input:** 2D spatial coordinates (x, y).
- **Hidden Layers:** 4 layers, each with 48 neurons and a nonlinear activation.
- **Output:** 3 values per point: u (velocity x), v (velocity y), p (pressure).

### ASCII Diagram

```
(x, y)
  │
  ▼
┌───────────────┐
│ Dense (48)    │  ← Hidden Layer 1 (tanh/swish/mish)
└───────────────┘
  │
  ▼
┌───────────────┐
│ Dense (48)    │  ← Hidden Layer 2 (tanh/swish/mish)
└───────────────┘
  │
  ▼
┌───────────────┐
│ Dense (48)    │  ← Hidden Layer 3 (tanh/swish/mish)
└───────────────┘
  │
  ▼
┌───────────────┐
│ Dense (48)    │  ← Hidden Layer 4 (tanh/swish/mish)
└───────────────┘
  │
  ▼
┌───────────────┐
│ Dense (3)     │  ← Output Layer (linear)
└───────────────┘
  │
  ▼
(u, v, p)
```

### Notes

- **Activation Functions:** The hidden layers use a nonlinear activation (`tanh` by default, but `swish` and `mish` are also available for experimentation).
- **Output Layer:** No activation (linear), as the network must be able to predict both positive and negative values for velocity and pressure.
- **Why this design?** The depth and width are chosen to balance expressiveness (ability to capture complex flow features) and trainability (avoiding overfitting and vanishing gradients).

---

## 3. Physics-Informed Loss Construction (`pinn.py`)

### 3.1. What makes it "physics-informed"?

- **Key idea:** Instead of training the network to fit data, we train it to satisfy the governing equations (Navier-Stokes) and boundary conditions everywhere in the domain.
- **How:** The loss function is constructed so that the network's outputs (u, v, p) must satisfy:
  - The Navier-Stokes equations at randomly sampled points in the domain (collocation points).
  - The boundary conditions at randomly sampled points on the domain boundaries and the cylinder surface.

### 3.2. Loss Terms (as implemented in `pinn.py`)

#### **A. PDE Residual Loss (Interior/Collocation Points)**

For each collocation point \((x, y)\), the following residuals are computed using automatic differentiation:

```math
\begin{align*}
R_u &= u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{1}{\rho} \frac{\partial p}{\partial x} - \frac{\mu}{\rho} \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) \\
R_v &= u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{1}{\rho} \frac{\partial p}{\partial y} - \frac{\mu}{\rho} \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) \\
R_c &= \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}
\end{align*}
```

- These are concatenated as `[R_u, R_v, R_c]` and the loss is typically the mean squared error (MSE) or log-cosh of these residuals over all collocation points.
- **Purpose:** Forces the network to satisfy the steady-state Navier-Stokes equations everywhere in the domain except the cylinder.

#### **B. Boundary Condition Losses**

For each boundary, the loss is constructed to enforce the physical boundary conditions:

- **Inlet (e.g., \(x=0\)):**
  - Enforce prescribed velocity profile, e.g.:
    ```math
    u(x=0, y) = 4y(1-y), \quad v(x=0, y) = 0
    ```
  - In code, this is implemented as:
    ```python
    uv_in = tf.concat([u_inn[0], v_inn[0], u_inn[0]], axis=-1)
    ```
    where the first and third components are set to the inlet profile, and the second (v) is set to zero.

- **Outlet (e.g., \(x=2\)):**
  - Often enforces pressure or Neumann condition. In the code:
    ```python
    uv_out = tf.concat([p_r[0], p_r[0], p_r[0]], axis=-1)
    ```
    This typically means the outlet loss is constructed to enforce a pressure condition.

- **Walls (top/bottom, \(y=0, y=1\)):**
  - No-slip: \(u=0, v=0\)
    ```python
    uv_w1 = tf.concat([u_grads_l[0], v_grads_l[0], p_l[2]], axis=-1)
    uv_w2 = tf.concat([u_grads_l[0], v_grads_l[0], p_l[2]], axis=-1)
    ```
    The first two components enforce \(u=0, v=0\).

- **Cylinder:**
  - No-slip: \(u=0, v=0\)
    ```python
    uv_circle = tf.concat([u_grads_l[0], v_grads_l[0], u_grads_l[0]], axis=-1)
    ```
    Both \(u\) and \(v\) are enforced to be zero on the cylinder surface.

#### **C. Total Loss**

The total loss is the sum of the losses for each output:
- **PDE residual loss:** Enforces the Navier-Stokes equations in the domain.
- **Boundary losses:** Enforce the boundary conditions at inlet, outlet, walls, and cylinder.

Each output of the model corresponds to a different region or boundary, and the loss for each is typically computed as the mean squared error (or log-cosh) between the predicted and target values (often zeros for residuals and no-slip boundaries).

### 3.3. Why this approach?

- **No labeled data required:** The network learns the solution by satisfying the physics, not by fitting to precomputed data.
- **Generalization:** The solution is valid everywhere in the domain, not just at discrete points.
- **Automatic differentiation:** Enables exact computation of derivatives needed for PDE residuals, avoiding numerical errors from finite differences.

---

## 4. Training Process (`main.py`, `optimizer.py`)

### 4.1. Data Preparation

- **Collocation Points:** Randomly sampled points in the domain (excluding the cylinder) for enforcing the PDE.
- **Boundary Points:** Randomly sampled points on each boundary (inlet, outlet, walls, cylinder surface) for enforcing boundary conditions.
- **Why random sampling?** Ensures the network learns the solution everywhere, not just on a grid. This is called "mesh-free" training.

### 4.2. Optimizer

- **L-BFGS-B:** A quasi-Newton optimizer, well-suited for PINNs because:
  - It can handle the complex, non-convex loss landscape.
  - It converges efficiently for small-to-medium-sized networks.
  - It uses both the loss and its gradients (computed via automatic differentiation).
- **Why not Adam?** Adam is often used for initial training, but L-BFGS-B can achieve higher accuracy for PINNs due to its second-order nature.

### 4.3. Training Loop

- The optimizer updates the network weights to minimize the total loss (physics + boundary).
- At each step, the loss and its gradient with respect to the weights are computed using TensorFlow's automatic differentiation.
- Loss history is recorded for analysis.

### 4.4. Model Inputs and Outputs

- **Inputs:** List of arrays, each corresponding to a set of points (collocation, inlet, outlet, walls, cylinder).
- **Outputs:** List of arrays, each corresponding to the residuals or boundary condition errors at those points.
- **Why this design?** Keras models can have multiple inputs/outputs, allowing the PINN to handle different loss terms for different regions/boundaries in a modular way.

---

## 5. Gradient Layer (`layer.py`)

- **Purpose:** Computes all necessary derivatives of the network outputs with respect to inputs using TensorFlow's `GradientTape`.
- **Why?** Needed for evaluating the PDE residuals and enforcing the physics in the loss function.
- **How?** For each input point, the layer computes:
  - First and second derivatives of u and v with respect to x and y.
  - First derivatives of p with respect to x and y.

---

## 6. Visualization and Evaluation (`main.py`)

- After training, the network is evaluated on a meshgrid of (x, y) points to produce fields for u, v, and p.
- Contour plots are generated to visualize the predicted velocity and pressure fields.
- Velocity profiles along specified cross-sections are plotted to analyze flow characteristics (e.g., boundary layer, wake).

---

## 7. Summary Diagram

```
(x, y) ──► [Dense 48, tanh] ──► [Dense 48, tanh] ──► [Dense 48, tanh] ──► [Dense 48, tanh] ──► (u, v, p)
      │
      ├─ Collocation points: Enforce PDE residuals (Navier-Stokes, continuity)
      ├─ Boundary points: Enforce boundary conditions (inlet, outlet, walls, cylinder)
      │
      ▼
  Physics-Informed Loss (automatic differentiation)
      │
      ▼
  L-BFGS-B optimizer (updates weights)
```

---

## 8. Why is this architecture effective?

- **Mesh-free:** No need for mesh generation; works with scattered points.
- **Flexible:** Can handle complex geometries and boundary conditions.
- **Physics-driven:** Solution is guaranteed to satisfy the governing equations and boundary conditions (to within the optimizer's tolerance).
- **Generalizable:** Once trained, the network can predict (u, v, p) at any (x, y) in the domain.
- **Automatic differentiation:** Provides accurate gradients for PDE residuals, crucial for enforcing physics.

---

## 9. How the Code Components Fit Together

- **`network.py`:** Defines the neural network architecture and custom activations.
- **`pinn.py`:** Wraps the network with physics-informed loss construction, using the `GradientLayer` for derivatives.
- **`optimizer.py`:** Implements the L-BFGS-B optimizer for efficient training of the PINN.
- **`main.py`:** Prepares data, builds and trains the model, and visualizes results.

---

## 10. References

- [Physics-Informed Neural Networks (PINNs) - Raissi et al.](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [PINN Review Paper](https://arxiv.org/abs/2107.09443)
- [L-BFGS-B Algorithm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html)

---

