import scipy.optimize
import numpy as np
import tensorflow as tf
import tqdm

class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.
    Attributes:
        model: optimization target model.
        samples: training samples.
        factr: function convergence condition. typical values for factr are: 1e12 for low accuracy;
               1e7 for moderate accuracy; 10.0 for extremely high accuracy.
        pgtol: gradient convergence condition.
        m: maximum number of variable metric corrections used to define the limited memory matrix.
        maxls: maximum number of line search steps (per iteration).
        maxiter: maximum number of iterations.
        metris: log metrics
        progbar: progress bar
    """

    def __init__(self, model, x_train, y_train, factr=1e5, m=50, maxls=50, maxiter=30000, loss_weights=None):
        """
        Args:
            model: optimization target model.Sw
            samples: training samples.
            factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
                   1e7 for moderate accuracy; 10.0 for extremely high accuracy.
            pgtol: gradient convergence condition.
            m: maximum number of variable metric corrections used to define the limited memory matrix.
            maxls: maximum number of line search steps (per iteration).
            maxiter: maximum number of iterations.
        """

        # set attributes
        self.model = model
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.loss_history = []  # Custom list to store loss values
        self.individual_loss_history = []  # Custom list to store component losses
        self.smoothed_losses = None  # for exponential moving average
        self.alpha = 0.3  # smoothing factor
        self.weight_clip = (0.05, 0.5)  # clip weights between these values
        # Add loss_weights
        if loss_weights is None:
            self.loss_weights = [1.0] * len(self.x_train)
        else:
            self.loss_weights = loss_weights

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

    def update_loss_weights(self):
        """
        Robust dynamic weighting:
        - smooths recent losses with EMA
        - computes inverse magnitude weights
        - clips weights to prevent instability
        """
        if not self.individual_loss_history:
            return

        recent_losses = np.array(self.individual_loss_history[-1])
        recent_losses = np.clip(recent_losses, 1e-8, None)  # avoid div by zero

        if self.smoothed_losses is None:
            self.smoothed_losses = recent_losses
        else:
            self.smoothed_losses = (
                self.alpha * recent_losses + (1 - self.alpha) * self.smoothed_losses
            )

        inv_losses = 1.0 / self.smoothed_losses
        weights = inv_losses / np.sum(inv_losses)

        # Clip to avoid overemphasis
        min_w, max_w = self.weight_clip
        weights = np.clip(weights, min_w, max_w)

        # Normalize again after clipping
        weights /= np.sum(weights)

        self.loss_weights = weights.tolist()
        print(f"[Loss Weights Updated] {np.round(self.loss_weights, 4)}")

    @tf.function
    def tf_evaluate(self, x, y):
        """
        Evaluate weighted loss and gradients for weights as tf.Tensor.
        Returns total weighted loss and list of individual losses.
        """
        with tf.GradientTape() as g:
            individual_losses = []
            total_loss = 0.0
            for i in range(len(x)):
                model_inputs = []
                for j in range(len(x)):
                    if i == j:
                        model_inputs.append(x[j])
                    else:
                        model_inputs.append(tf.zeros_like(x[j]))
                model_output = self.model(model_inputs)[i]
                loss_i = tf.reduce_mean(tf.keras.losses.logcosh(model_output, y[i]))
                individual_losses.append(loss_i)
                total_loss += self.loss_weights[i] * loss_i
            loss = total_loss
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads, individual_losses

    def evaluate(self, weights):
        self.set_weights(weights)
        loss, grads, individual_losses = self.tf_evaluate(self.x_train, self.y_train)
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')
        individual_losses = [l.numpy().astype('float64') for l in individual_losses]
        return loss, grads, individual_losses

    def callback(self, weights):
        loss, _, individual_losses = self.evaluate(weights)
        self.loss_history.append(loss)
        self.individual_loss_history.append(individual_losses)

        # Rebalance every 50 iterations (adjustable)
        if len(self.loss_history) % 50 == 0:
            self.update_loss_weights()


    def fit(self):
        """
        Train the model using L-BFGS-B algorithm.
        """
        initial_weights = np.concatenate(
            [w.flatten() for w in self.model.get_weights()])
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))

        # Use tqdm for progress bar
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

    # Function to save the model
    def save_model(self, filepath):
        """
        Save the model to a file.
        Args:
            filepath: path to save the model.
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")