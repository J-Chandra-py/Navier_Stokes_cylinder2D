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