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
        del ggg

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
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

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


"""tf_silent.py"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



"""main.py"""
# Example: Flexible training loop (not meant to be run as-is)
# Sample as many points as you want for each region:
# xyt_interior, xyt_boundary, xyt_cylinder, etc.

# Example: define your own loss function using any number of points per region
def total_loss():
    loss = 0.0
    # Interior PDE loss
    loss += pde_loss(pinn, tf.constant(xyt_interior, dtype=tf.float32))
    # Inlet BC loss
    loss += boundary_loss(pinn, tf.constant(xyt_inlet, dtype=tf.float32), tf.constant(target_inlet, dtype=tf.float32))
    # Wall BC loss
    loss += boundary_loss(pinn, tf.constant(xyt_wall, dtype=tf.float32), tf.constant(target_wall, dtype=tf.float32))
    # Cylinder BC loss
    loss += boundary_loss(pinn, tf.constant(xyt_cylinder, dtype=tf.float32), tf.constant(target_cylinder, dtype=tf.float32))
    # ...add more as needed
    return loss

# Now you can sample any number of points for each region, and only those regions you want.
# The optimizer only needs the total_loss function.

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
        return self.network(xy)

    def predict_uv(self, xy):
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
        """
        Compute the curl of the momentum equation (vorticity transport equation).
        Returns: residual (N, 1)
        """
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
            # Compute vorticity: ω = v_x - u_y
            omega = v_x - u_y
            # Compute the vorticity transport equation residual:
            # ∂ω/∂t + u ∂ω/∂x + v ∂ω/∂y = ν ∇²ω (steady, so ∂ω/∂t = 0)
            omega_x = tape1.gradient(omega, xy)[:, 0:1]
            omega_y = tape1.gradient(omega, xy)[:, 1:2]
            lap_omega = tape2.gradient(omega_x, xy)[:, 0:1] + tape2.gradient(omega_y, xy)[:, 1:2]
            eqm_res = u * omega_x + v * omega_y - self.nu * lap_omega
        del tape1, tape2
        return eqm_res

    def pmpg_residual(self, xy):
        """
        Compute the squared norm of the pressure gradient (PMPG) surrogate:
        PMPG = ((u·∇)u - ν∇²u)^2 + ((u·∇)v - ν∇²v)^2
        """
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
        """
        Compute boundary residuals (penalty for BCs).
        If normal/tangent are provided, can enforce u·n=0 and (u-u_wall)·t=0.
        Otherwise, just penalize (u-u_target), (v-v_target).
        """
        u, v = self.predict_uv(xy)
        res = 0.0
        if enforce_normal and normal is not None:
            n_x, n_y = normal
            res += tf.square(u * n_x + v * n_y)
        if enforce_tangent and tangent is not None:
            t_x, t_y = tangent
            res += tf.square((u - u_target) * t_x + (v - v_target) * t_y)
        if not enforce_normal and not enforce_tangent:
            res += tf.square(u - u_target) + tf.square(v - v_target)
        return tf.reduce_mean(res)


# Example: Loss function using penalty method and streamfunction PINN

def total_loss():
    λ1 = 1.0   # boundary penalty
    λ2 = 1e6   # equilibrium penalty
    λ3 = 1.0   # PMPG penalty

    # PMPG loss (pressure gradient surrogate)
    loss_pmpg = tf.reduce_mean(pinn.pmpg_residual(tf.constant(xyt_interior, dtype=tf.float32)))
    # Equilibrium (vorticity transport) loss
    loss_eqm = tf.reduce_mean(tf.square(pinn.equilibrium_residual(tf.constant(xyt_interior, dtype=tf.float32))))
    # Boundary loss (example: Dirichlet BCs)
    loss_bc = pinn.boundary_residual(tf.constant(xyt_wall, dtype=tf.float32),
                                     tf.constant(u_wall, dtype=tf.float32),
                                     tf.constant(v_wall, dtype=tf.float32),
                                     normal=wall_normal, tangent=wall_tangent,
                                     enforce_normal=True, enforce_tangent=True)
    # ...add more BCs as needed...

    return λ1 * loss_bc + λ2 * loss_eqm + λ3 * loss_pmpg

# Example usage:
# network = Network().build(num_inputs=2, layers=[48,48,48,48], activation='tanh', num_outputs=1)  # output ψ
# pinn = StreamFunctionPINN(network, rho=1.0, nu=0.01)
# lbfgs = L_BFGS_B(model=network, loss_fn=total_loss, factr=1e5, m=50, maxls=50, maxiter=30000)
# lbfgs.fit()
# lbfgs = L_BFGS_B(model=network, loss_fn=total_loss, factr=1e5, m=50, maxls=50, maxiter=30000)
# lbfgs.fit()
