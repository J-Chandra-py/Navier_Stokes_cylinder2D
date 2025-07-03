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
        # Add loss_weights
        if loss_weights is None:
            self.loss_weights = [1.0] * len(self.x_train)
        else:
            self.loss_weights = loss_weights
        # initialize the progress bar
        # self.progbar = tf.keras.callbacks.ProgbarLogger(
        #     count_mode='steps', stateful_metrics=self.metrics)
        # self.progbar.set_params( {
        #     'verbose':1, 'epochs':1, 'steps':self.maxiter, 'metrics':self.metrics})

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
    def tf_evaluate(self, x, y):
        """
        Evaluate weighted loss and gradients for weights as tf.Tensor.
        Args:
            x: list of input data.
            y: list of target data.
        Returns:
            weighted loss and gradients for weights as tf.Tensor.
        """
        with tf.GradientTape() as g:
            total_loss = 0.0
            for i in range(len(x)):
                # Prepare input list: only i-th is real, others are zeros
                model_inputs = []
                for j in range(len(x)):
                    if i == j:
                        model_inputs.append(x[j])
                    else:
                        # Use zeros with the same shape as x[j]
                        model_inputs.append(tf.zeros_like(x[j]))
                # Model output is a list; select i-th output
                model_output = self.model(model_inputs)[i]
                loss_i = tf.reduce_mean(tf.keras.losses.logcosh(model_output, y[i]))
                total_loss += self.loss_weights[i] * loss_i
            loss = total_loss
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
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
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
        # self.progbar.on_batch_begin(0)
        loss, _ = self.evaluate(weights)
        self.loss_history.append(loss)  # Store the loss
        # self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))

    # def fit(self):
    #     """
    #     Train the model using L-BFGS-B algorithm.
    #     """

    #     # get initial weights as a flat vector
    #     initial_weights = np.concatenate(
    #         [ w.flatten() for w in self.model.get_weights() ])
    #     # optimize the weight vector
    #     print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
    #     self.progbar.on_train_begin()
    #     self.progbar.on_epoch_begin(1)
    #     scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
    #         factr=self.factr, m=self.m,
    #         maxls=self.maxls, maxiter=self.maxiter, callback=self.callback)
    #     self.progbar.on_epoch_end(1)
    #     self.progbar.on_train_end()

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