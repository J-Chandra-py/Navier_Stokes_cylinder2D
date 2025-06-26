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
