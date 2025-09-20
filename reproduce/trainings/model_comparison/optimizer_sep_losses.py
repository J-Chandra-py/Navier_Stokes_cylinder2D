import scipy.optimize
import numpy as np
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt

class TrainingTracker:
    """
    Tracks losses, gradients, weights, and component losses during optimization.
    """
    def __init__(self):
        self.losses = []
        self.grads = []
        self.weights = []
        self.component_losses = []

    def record(self, loss, grads, weights, component_losses=None):
        self.losses.append(loss)
        self.grads.append(np.copy(grads))
        self.weights.append(np.copy(weights))
        if component_losses is not None:
            # Store as a list of floats for each iteration
            self.component_losses.append([float(cl) for cl in component_losses])
        else:
            self.component_losses.append(None)

    def plot(self):
        """
        Plot loss, gradient norm, weight norm, and component losses (all log-scaled).
        """
        ncols = 4 if any(self.component_losses) else 3
        plt.figure(figsize=(16, 4) if ncols == 4 else (12, 4))
        plt.subplot(1, ncols, 1)
        plt.plot(self.losses)
        plt.yscale('log')
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.subplot(1, ncols, 2)
        grad_norms = [np.linalg.norm(g) for g in self.grads]
        plt.plot(grad_norms)
        plt.yscale('log')
        plt.title('Gradient Norm')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')

        plt.subplot(1, ncols, 3)
        weight_norms = [np.linalg.norm(w) for w in self.weights]
        plt.plot(weight_norms)
        plt.yscale('log')
        plt.title('Weight Norm')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')

        if ncols == 4:
            plt.subplot(1, ncols, 4)
            # Only plot if component_losses are available and not None
            comp_losses = [cl for cl in self.component_losses if cl is not None]
            if comp_losses:
                comp_losses_arr = np.array(comp_losses)
                for i in range(comp_losses_arr.shape[1]):
                    plt.plot(comp_losses_arr[:, i], label=f'Comp {i}')
                plt.yscale('log')
                plt.title('Component Losses')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.
    """

    def __init__(self, model, x_train, y_train, loss_weights=None, factr=1e5, m=50, maxls=50, maxiter=30000):
        """
        Args:
            model: optimization target model.
            x_train: list of training input arrays.
            y_train: list of training output arrays.
            loss_weights: list of weights for each component loss (optional).
            factr: convergence condition (see scipy.optimize.fmin_l_bfgs_b).
            m: maximum number of variable metric corrections.
            maxls: maximum number of line search steps (per iteration).
            maxiter: maximum number of iterations.
        """
        self.model = model
        self.x_train = [tf.constant(x, dtype=tf.float32) for x in x_train]
        self.y_train = [tf.constant(y, dtype=tf.float32) for y in y_train]
        self.loss_weights = loss_weights if loss_weights is not None else [1.0] * len(self.x_train)
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.loss_history = []
        self.tracker = TrainingTracker()

    def set_weights(self, flat_weights):
        """
        Set weights to the model from a flat vector.
        """
        shapes = [w.shape for w in self.model.get_weights()]
        split_ids = np.cumsum([np.prod(shape) for shape in [0] + shapes])
        weights = [flat_weights[from_id:to_id].reshape(shape)
                   for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes)]
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        """
        Evaluate loss and gradients for weights as tf.Tensor.
        Returns total loss, gradients, and component losses.
        """
        with tf.GradientTape() as g:
            y_pred = self.model(x)  # This is a list of outputs
            component_losses = [
                tf.reduce_mean(tf.keras.losses.logcosh(y_true_i, y_pred_i))
                for y_true_i, y_pred_i in zip(y, y_pred)
            ]
            loss = tf.add_n([cl * lw for cl, lw in zip(component_losses, self.loss_weights)])
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads, component_losses

    def evaluate(self, weights):
        """
        Evaluate loss and gradients for weights as ndarray.
        """
        self.set_weights(weights)
        loss, grads, component_losses = self.tf_evaluate(self.x_train, self.y_train)
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')
        comp_losses = [cl.numpy().astype('float64') for cl in component_losses]
        self.tracker.record(loss, grads, weights, comp_losses)
        return loss, grads

    def callback(self, weights):
        """
        Callback that records the loss for each iteration.
        """
        loss, _ = self.evaluate(weights)
        self.loss_history.append(loss)

    def plot_training(self):
        """
        Visualize tracked losses, gradients, weights, and component losses.
        """
        self.tracker.plot()

    def fit(self):
        """
        Train the model using L-BFGS-B algorithm.
        """
        initial_weights = np.concatenate([w.flatten() for w in self.model.get_weights()])
        print(f'Optimizer: L-BFGS-B (maxiter={self.maxiter})')
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