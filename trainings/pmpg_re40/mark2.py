"""network.py"""

import tensorflow as tf

class Network:
    """
    Build a physics informed neural network (PINN) model for the steady Navier-Stokes equations.
    Attributes:
        activations: custom activation functions.
    """

    def __init__(self):
        """
        Setup custom activation functions.
        """
        self.activations = {
            'tanh' : 'tanh',
            'swish': self.swish,
            'mish' : self.mish,
        }
    # Examples of other activation functions:
    def swish(self, x):
        """
        Swish activation function.
        Args:
            x: activation input.
        Returns:
            Swish output.
        """
        return x * tf.math.sigmoid(x)

    def mish(self, x):
        """
        Mish activation function.
        Args:
            x: activation input.
        Returns:
            Mish output.
        """
        return x * tf.math.tanh(tf.softplus(x))

    def build(self, num_inputs=2, layers=[48,48,48,48], activation='tanh', num_outputs=3):
        """
        Build a PINN model for the steady Navier-Stokes equation with input shape (x,y) and output shape (u, v, p).
        Args:
            num_inputs: number of input variables. Default is 2 for (x, y).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default is 3 for (u, v, p).
        Returns:
            keras network model
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        # hidden layers
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=self.activations[activation],
                kernel_initializer='he_normal')(x)
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='he_normal')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
"""layer.py"""
import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute derivatives for the steady Navier-Stokes equation.
    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, xyt):
        """
        Computing derivatives for the steady Navier-Stokes equation.
        Args:
            xy: input variable.
        Returns:
            psi: stream function.
            p_grads: pressure and its gradients.
            u_grads: u and its gradients.
            v_grads: v and its gradients.
        """

        # For 1D outputs, tf.gradients is more efficient than batch_jacobian
        x, y = [xyt[..., i, tf.newaxis] for i in range(xyt.shape[-1])]
        with tf.GradientTape(persistent=True) as g2:
            g2.watch([x, y])
            with tf.GradientTape(persistent=True) as g1:
                g1.watch([x, y])

                u_v_p = self.model(tf.concat([x, y], axis=-1))
                u = u_v_p[..., 0, tf.newaxis]
                v = u_v_p[..., 1, tf.newaxis]
                p = u_v_p[..., 2, tf.newaxis]
            u_x = g1.gradient(u, x)
            v_x = g1.gradient(v, x)
            u_y = g1.gradient(u, y)
            v_y = g1.gradient(v, y)
            p_x = g1.gradient(p, x)
            p_y = g1.gradient(p, y)
            del g1
        u_xx = g2.gradient(u_x, x)
        u_yy = g2.gradient(u_y, y)

        v_xx = g2.gradient(v_x, x)
        v_yy = g2.gradient(v_y, y)
        del g2
        # if more derivatives are required...
        # del ggg

        p_grads = p, p_x, p_y
        u_grads = u, u_x, u_y, u_xx, u_yy
        v_grads = v, v_x, v_y, v_xx, v_yy

        return p_grads, u_grads, v_grads

"""pinn.py"""
import tensorflow as tf
from layer import GradientLayer
import numpy as np

class PINN:
    """
    Physics Informed Neural Network for steady Navier-Stokes.
    Predicts (u, v, p) at (x, y). Losses for PDE, BC, etc., are computed outside this class.
    """

    def __init__(self, network, rho=1, mu=0.01):
        self.network = network
        self.rho = rho
        self.mu = mu
        self.grads = GradientLayer(self.network)

    def predict(self, xy):
        """
        Predict u, v, p at given (x, y) points.
        Args:
            xy: numpy array or tf.Tensor of shape (N, 2)
        Returns:
            u, v, p: tensors of shape (N, 1)
        """
        preds = self.network(xy)
        u = preds[..., 0:1]
        v = preds[..., 1:2]
        p = preds[..., 2:3]
        # Optionally return numpy if input is numpy
        if isinstance(xy, np.ndarray):
            return u.numpy(), v.numpy(), p.numpy()
        return u, v, p

    def get_pde_residuals(self, xy):
        """
        Compute PDE residuals at given (x, y) points.
        Args:
            xy: tf.Tensor of shape (N, 2)
        Returns:
            residuals: tf.Tensor of shape (N, 3)
        """
        p_grads, u_grads, v_grads = self.grads(xy)
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads
        u_eqn =  u*u_x + v*u_y + p_x/self.rho - self.mu*(u_xx + u_yy) / self.rho
        v_eqn =  u*v_x + v*v_y + p_y/self.rho - self.mu*(v_xx + v_yy) / self.rho
        uv_eqn = u_x + v_y
        return tf.concat([u_eqn, v_eqn, uv_eqn], axis=-1)

# Utility functions for loss computation (outside the model)
def pde_loss(pinn, xy):
    res = pinn.get_pde_residuals(xy)
    return tf.reduce_mean(tf.square(res))

def boundary_loss(pinn, xy, target_uvp, mask=None):
    u, v, p = pinn.predict(xy)
    pred = tf.concat([u, v, p], axis=-1)
    loss = tf.square(pred - target_uvp)
    if mask is not None:
        loss = tf.boolean_mask(loss, mask)
    return tf.reduce_mean(loss)

def no_penetration_loss(pinn, xy, normal):
    u, v, _ = pinn.predict(xy)
    n_x, n_y = normal  # shape (N, 1) each
    return tf.reduce_mean(tf.square(u * n_x + v * n_y))

"""streamfunction_pinn.py"""
import tensorflow as tf

class StreamFunctionPINN:
    """
    PINN for steady incompressible Navier-Stokes using streamfunction formulation.
    The network predicts ψ(x, y). Velocities are u = dψ/dy, v = -dψ/dx.
    """

    def __init__(self, network, rho=1.0, nu=0.01):
        self.network = network
        self.rho = rho
        self.nu = nu

    def predict_psi(self, xy):
        # Ensure input is a tensor
        xy = tf.convert_to_tensor(xy, dtype=tf.float32)
        return self.network(xy)

    def predict_uv(self, xy):
        xy = tf.convert_to_tensor(xy, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xy)
            psi = self.network(xy)
            psi_x = tape.gradient(psi, xy)[:, 0:1]
            psi_y = tape.gradient(psi, xy)[:, 1:2]
        u = psi_y
        v = -psi_x
        del tape
        return u, v

    def equilibrium_residual(self, xy):
        xy = tf.convert_to_tensor(xy, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xy)
                psi = self.network(xy)
                psi_x = tape1.gradient(psi, xy)[:, 0:1]
                psi_y = tape1.gradient(psi, xy)[:, 1:2]
                u = psi_y
                v = -psi_x
            u_x = tape1.gradient(u, xy)[:, 0:1]
            u_y = tape1.gradient(u, xy)[:, 1:2]
            v_x = tape1.gradient(v, xy)[:, 0:1]
            v_y = tape1.gradient(v, xy)[:, 1:2]
            # Compute vorticity: ω = v_x - u_y
            omega = v_x - u_y
            # Compute gradients of omega
            omega_grad = tape1.gradient(omega, xy)
            if omega_grad is None:
                # Return zeros if gradient is not available (should not happen in training)
                omega_x = tf.zeros_like(omega)
                omega_y = tf.zeros_like(omega)
            else:
                omega_x = omega_grad[:, 0:1]
                omega_y = omega_grad[:, 1:2]
            lap_omega_x = tape2.gradient(omega_x, xy)
            lap_omega_y = tape2.gradient(omega_y, xy)
            if lap_omega_x is None or lap_omega_y is None:
                lap_omega = tf.zeros_like(omega)
            else:
                lap_omega = lap_omega_x[:, 0:1] + lap_omega_y[:, 1:2]
            eqm_res = u * omega_x + v * omega_y - self.nu * lap_omega
        del tape1, tape2
        return eqm_res

    def pmpg_residual(self, xy):
        xy = tf.convert_to_tensor(xy, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xy)
                psi = self.network(xy)
                psi_x = tape1.gradient(psi, xy)[:, 0:1]
                psi_y = tape1.gradient(psi, xy)[:, 1:2]
                u = psi_y
                v = -psi_x
            u_x = tape1.gradient(u, xy)[:, 0:1]
            u_y = tape1.gradient(u, xy)[:, 1:2]
            v_x = tape1.gradient(v, xy)[:, 0:1]
            v_y = tape1.gradient(v, xy)[:, 1:2]
            lap_u = tape2.gradient(u_x, xy)[:, 0:1] + tape2.gradient(u_y, xy)[:, 1:2]
            lap_v = tape2.gradient(v_x, xy)[:, 0:1] + tape2.gradient(v_y, xy)[:, 1:2]
            adv_u = u * u_x + v * u_y
            adv_v = u * v_x + v * v_y
            pmpg_u = adv_u - self.nu * lap_u
            pmpg_v = adv_v - self.nu * lap_v
            pmpg = tf.square(pmpg_u) + tf.square(pmpg_v)
        del tape1, tape2
        return pmpg

    def boundary_residual(self, xy, u_target, v_target, normal=None, tangent=None, enforce_normal=True, enforce_tangent=True):
        xy = tf.convert_to_tensor(xy, dtype=tf.float32)
        u_target = tf.convert_to_tensor(u_target, dtype=tf.float32)
        v_target = tf.convert_to_tensor(v_target, dtype=tf.float32)
        u, v = self.predict_uv(xy)
        res = 0.0
        if enforce_normal and normal is not None:
            n_x, n_y = normal
            n_x = tf.convert_to_tensor(n_x, dtype=tf.float32)
            n_y = tf.convert_to_tensor(n_y, dtype=tf.float32)
            res += tf.square(u * n_x + v * n_y)
        if enforce_tangent and tangent is not None:
            t_x, t_y = tangent
            t_x = tf.convert_to_tensor(t_x, dtype=tf.float32)
            t_y = tf.convert_to_tensor(t_y, dtype=tf.float32)
            res += tf.square((u - u_target) * t_x + (v - v_target) * t_y)
        if not enforce_normal and not enforce_tangent:
            res += tf.square(u - u_target) + tf.square(v - v_target)
        return tf.reduce_mean(res)
        return tf.reduce_mean(res)

"""optimizer.py"""
import scipy.optimize
import numpy as np
import tensorflow as tf
import tqdm

class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.
    Now expects a loss function (callable) that computes the total loss.
    """
    def __init__(self, model, loss_fn, factr=1e5, m=50, maxls=50, maxiter=30000):
        self.model = model
        self.loss_fn = loss_fn
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.loss_history = []

    def set_weights(self, flat_weights):
        """
        Set weights to the model.
        Args:
            flat_weights: flatten weights.
        """

        # get model weights
        shapes = [ w.shape for w in self.model.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, weights):
        """
        Evaluate loss and gradients for weights as tf.Tensor.
        Args:
            x: input data.
        Returns:
            loss and gradients for weights as tf.Tensor.
        """

        with tf.GradientTape() as g:
            # LOSS FUNCTION IS SPECIFIED HERE:
            loss = self.loss_fn()
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):
        """
        Evaluate loss and gradients for weights as ndarray.
        Args:
            weights: flatten weights.
        Returns:
            loss and gradients for weights as ndarray.
        """

        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads = self.tf_evaluate(weights)
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        # Filter out None gradients (can happen if some variables are not used in loss)
        grads = [g for g in grads if g is not None]
        if not grads:
            raise RuntimeError("All gradients are None. Check your loss function and model.")
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')

        return loss, grads

    def callback(self, weights):
        """
        Callback that records the loss for each iteration.
        Args:
            weights: flatten weights.
        """
        loss, _ = self.evaluate(weights)
        self.loss_history.append(loss)

    def fit(self):
        """
        Train the model using L-BFGS-B algorithm.
        """
        initial_weights = np.concatenate(
            [w.flatten() for w in self.model.get_weights()])
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.loss_history = []
        pbar = tqdm.tqdm(total=self.maxiter, desc="L-BFGS-B", unit="iter")
        def callback_with_bar(weights):
            self.callback(weights)
            pbar.update(1)
            pbar.set_postfix(loss=self.loss_history[-1] if self.loss_history else None)
        scipy.optimize.fmin_l_bfgs_b(
            func=self.evaluate,
            x0=initial_weights,
            factr=self.factr,
            m=self.m,
            maxls=self.maxls,
            maxiter=self.maxiter,
            callback=callback_with_bar
        )
        pbar.close()



import tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from network import Network
from optimizer import L_BFGS_B
from streamfunction_pinn import StreamFunctionPINN



# imitate flow behavior at pipe inlet, zero at edges & max around center
def u_0(xy):
    """
    Initial wave form.
    Args:
        tx: variables (t, x) as tf.Tensor.
    Returns:
        u(t, x) as tf.Tensor.
    """

    x = xy[..., 0, None]
    y = xy[..., 1, None]


    return    4*y*(1 - y) 



def contour(x, y, z, title, levels=100):
    """
    Contour plot.
    Args:
        grid: plot position.
        x: x-array.
        y: y-array.
        z: z-array.
        title: title string.
        levels: number of contour lines.
    """
    from matplotlib.patches import Circle
    vmin = np.min(z)
    vmax = np.max(z)
    font1 = {'family':'serif','size':20}
    fig, ax = plt.subplots(figsize=(10, 8))
    cs1 = ax.contour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    cs2 = ax.contourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    circle = Circle((0.5, 0.5), 0.1, fc='black', zorder=10)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title(title, fontdict=font1)
    ax.set_xlabel("x", fontdict=font1)
    ax.set_ylabel("y", fontdict=font1)
    ax.tick_params(axis='both', which='major', labelsize=15)
    cbar = fig.colorbar(cs2, ax=ax, pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     """
#     Test the physics informed neural network (PINN) model
#     for the cavity flow governed by the steady Navier-Stokes equation.
#     """

# number of training samples
num_train_samples = 5000
num_cylinder_samples = 4000  # more points around cylinder
num_wake_samples = 4000      # more points in wake region
num_annulus_samples = 3000   # more points in annulus
num_test_samples = 200

# inlet flow velocity
u0 = 1
rho = 1
mu = 0.025
nu = mu / rho

# build a core network model for streamfunction (output: 1)
network = Network().build(num_inputs=2, layers=[48,48,48,48], activation='tanh', num_outputs=1)
network.summary()
# build a StreamFunctionPINN model
pinn = StreamFunctionPINN(network, rho=1, nu=0.025)


# Domain and circle data
x_f = 2
x_ini = 0
y_f = 1
y_ini = 0
Cx = 0.5
Cy = 0.5
a = 0.1
b = 0.1

# Cylinder boundary points (more points)
theta_cyl = np.linspace(0, 2*np.pi, num_cylinder_samples, endpoint=False)
xyt_circle = np.stack([Cx + a * np.cos(theta_cyl), Cy + b * np.sin(theta_cyl)], axis=-1)

# Interior domain sampling (excluding cylinder)
xyt_eqn = np.random.rand(num_train_samples, 2)
xyt_eqn[...,0] = (x_f - x_ini)*xyt_eqn[...,0] + x_ini
xyt_eqn[...,1] = (y_f - y_ini)*xyt_eqn[...,1] + y_ini
# Remove points inside the cylinder
mask = ((xyt_eqn[:, 0] - Cx)**2/a**2 + (xyt_eqn[:, 1] - Cy)**2/b**2) >= 1
xyt_eqn = xyt_eqn[mask]

# Annulus region (more points)
theta = np.random.uniform(0, 2*np.pi, num_annulus_samples)
r = np.sqrt(np.random.uniform(1, 6.25, num_annulus_samples))
x = Cx + a * r * np.cos(theta)
y = Cy + b * r * np.sin(theta)
xyt_annulus = np.stack([x, y], axis=1)

# Wake region (more points)
x_strip = np.random.uniform(0.55, 0.75, num_wake_samples)
y_strip = np.random.uniform(0.3, 0.7, num_wake_samples)
xyt_strip = np.stack([x_strip, y_strip], axis=1)
# Remove points inside the cylinder in the wake
mask_wake = ((xyt_strip[:, 0] - Cx)**2/a**2 + (xyt_strip[:, 1] - Cy)**2/b**2) >= 1
xyt_strip = xyt_strip[mask_wake]

# Combine all interior points
xyt_interior = np.concatenate([xyt_eqn, xyt_annulus, xyt_strip], axis=0)

# Boundary points
xyt_w1 = np.random.rand(num_train_samples, 2)
xyt_w1[..., 0] = (x_f - x_ini)*xyt_w1[...,0] + x_ini
xyt_w1[..., 1] = y_ini

xyt_w2 = np.random.rand(num_train_samples, 2)
xyt_w2[..., 0] = (x_f - x_ini)*xyt_w2[...,0] + x_ini
xyt_w2[..., 1] = y_f

xyt_out = np.random.rand(num_train_samples, 2)
xyt_out[..., 0] = x_f

xyt_in = np.random.rand(num_train_samples, 2)
xyt_in[..., 0] = x_ini

# Wall boundary (combine top and bottom)
xyt_wall = np.concatenate([xyt_w1, xyt_w2], axis=0)

# Inlet boundary
xyt_inlet = xyt_in

# Target values for boundary conditions
# For streamfunction, set psi=0 on cylinder and walls, and set inlet/outlet as needed
psi_wall = np.zeros((xyt_wall.shape[0], 1))
psi_cylinder = np.zeros((xyt_circle.shape[0], 1))
psi_inlet = np.zeros((xyt_inlet.shape[0], 1))  # adjust if needed

# For velocity BCs (for penalty), set u_target, v_target as needed
u_wall = np.zeros((xyt_wall.shape[0], 1))
v_wall = np.zeros((xyt_wall.shape[0], 1))
# For cylinder, no-slip
u_cylinder = np.zeros((xyt_circle.shape[0], 1))
v_cylinder = np.zeros((xyt_circle.shape[0], 1))
# For inlet, parabolic profile
u_inlet = u_0(xyt_inlet)
v_inlet = np.zeros((xyt_inlet.shape[0], 1))

# Normals and tangents for wall/cylinder (for no-penetration and no-slip)
# Example: for bottom wall, normal is (0,1), tangent is (1,0)
wall_normal = (np.zeros((xyt_wall.shape[0], 1)), np.ones((xyt_wall.shape[0], 1)))
wall_tangent = (np.ones((xyt_wall.shape[0], 1)), np.zeros((xyt_wall.shape[0], 1)))
# For cylinder, normal is radial
dx = xyt_circle[:, 0:1] - Cx
dy = xyt_circle[:, 1:2] - Cy
norm = np.sqrt(dx**2 + dy**2)
cyl_normal = (dx/norm, dy/norm)
# Tangent is perpendicular
cyl_tangent = (-dy/norm, dx/norm)


# Define the total loss with penalty method (keep in main file for flexibility)
def total_loss():
    λ_bc = 1.0
    λ_eqm = 1e6
    λ_pmpg = 1.0

    # PMPG loss (pressure gradient surrogate)
    loss_pmpg = tf.reduce_mean(pinn.pmpg_residual(tf.constant(xyt_interior, dtype=tf.float32)))
    # Equilibrium (vorticity transport) loss
    loss_eqm = tf.reduce_mean(tf.square(pinn.equilibrium_residual(tf.constant(xyt_interior, dtype=tf.float32))))
    # Wall boundary loss (no-slip and no-penetration)
    loss_wall = pinn.boundary_residual(tf.constant(xyt_wall, dtype=tf.float32),
                                        tf.constant(u_wall, dtype=tf.float32),
                                        tf.constant(v_wall, dtype=tf.float32),
                                        normal=wall_normal, tangent=wall_tangent,
                                        enforce_normal=True, enforce_tangent=True)
    # Cylinder boundary loss (no-slip and no-penetration)
    loss_cylinder = pinn.boundary_residual(tf.constant(xyt_circle, dtype=tf.float32),
                                            tf.constant(u_cylinder, dtype=tf.float32),
                                            tf.constant(v_cylinder, dtype=tf.float32),
                                            normal=cyl_normal, tangent=cyl_tangent,
                                            enforce_normal=True, enforce_tangent=True)
    # Inlet boundary loss (prescribed u, v)
    loss_inlet = pinn.boundary_residual(tf.constant(xyt_inlet, dtype=tf.float32),
                                        tf.constant(u_inlet, dtype=tf.float32),
                                        tf.constant(v_inlet, dtype=tf.float32),
                                        normal=None, tangent=None,
                                        enforce_normal=False, enforce_tangent=False)
    loss_bc = loss_wall + loss_cylinder + loss_inlet
    return λ_bc * loss_bc + λ_eqm * loss_eqm + λ_pmpg * loss_pmpg


#Train the model using L-BFGS-B algorithm
lbfgs = L_BFGS_B(model=network, loss_fn=total_loss)
lbfgs.fit()



# Prediction and plotting
x = np.linspace(x_ini, x_f, num_test_samples)
y = np.linspace(y_ini, y_f, num_test_samples)
x, y = np.meshgrid(x, y)
xy = np.stack([x.flatten(), y.flatten()], axis=-1)
u, v = pinn.predict_uv(tf.constant(xy, dtype=tf.float32))
u = u.numpy().reshape(x.shape)
v = v.numpy().reshape(x.shape)
# Optionally, plot psi as well
psi = network(xy).numpy().reshape(x.shape)

# # Plotting (no pressure, only u, v, psi)
contour(x, y, psi, 'Streamfunction ψ')
contour(x, y, u, 'u')
contour(x, y, v, 'v')


fig0, ax0 = plt.subplots(1, 1, figsize=(20,8))
cf0 = ax0.contourf(x, y, u, np.arange(-0.5, 1.1, .02),
                extend='both',cmap='rainbow')
cbar0 = plt.colorbar(cf0, )
plt.title("u", fontdict = font1)
plt.xlabel("x", fontdict = font1)
plt.ylabel("y", fontdict = font1)
ax0.add_patch(Circle((0.5, 0.5), 0.1,color="black"))
plt.tick_params(axis='both', which='major', labelsize=15)
cbar0.ax.tick_params(labelsize=15)
plt.show()

###########################

fig0, ax0 = plt.subplots(1, 1,figsize=(20,8))
cf0 = ax0.contourf(x, y, v, np.arange(-0.4, 0.4, .02),
                extend='both',cmap='rainbow')
cbar0 = plt.colorbar(cf0, pad=0.03, aspect=25, format='%.0e')
plt.title("v", fontdict = font1)
plt.xlabel("x", fontdict = font1)
plt.ylabel("y", fontdict = font1)
ax0.add_patch(Circle((0.5, 0.5), 0.1,color="black"))
plt.tick_params(axis='both', which='major', labelsize=15)
cbar0.ax.tick_params(labelsize=15)


# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(len(lbfgs.loss_history)), np.log(lbfgs.loss_history), label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations')
plt.legend()
plt.grid()
plt.savefig('loss_curve_Re40.png')
plt.show()



import matplotlib.pyplot as plt
from matplotlib.patches import Circle

font1 = {'family':'serif','size':20}

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
# Contourf for pressure (or any scalar field)
cf = ax.contourf(x, y, u, np.arange(-0.2, 1, .02), extend='both', cmap='rainbow')
cbar = plt.colorbar(cf, pad=0.03, aspect=25, format='%.0e')
# Streamlines for velocity field
strm = ax.streamplot(x, y, u, v, color='k', density=3, linewidth=0.25)
# Add cylinder
# ax.add_patch(Circle((0.5, 0.5), 0.1, color="black"))
plt.title("u with Streamlines (Re40)", fontdict=font1)
plt.xlabel("x", fontdict=font1)
plt.ylabel("y", fontdict=font1)
plt.tick_params(axis='both', which='major', labelsize=15)
cbar.ax.tick_params(labelsize=15)
plt.savefig('u_streamlines_Re40.png', dpi=300)
plt.show()