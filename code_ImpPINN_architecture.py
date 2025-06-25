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

        x, y = [ xyt[..., i, tf.newaxis] for i in range(xyt.shape[-1]) ]
        with tf.GradientTape(persistent=True) as ggg:
            ggg.watch(x)
            ggg.watch(y)

            with tf.GradientTape(persistent=True) as gg:
                gg.watch(x)
                gg.watch(y)

                with tf.GradientTape(persistent=True) as g:
                    g.watch(x)
                    g.watch(y)

                    u_v_p = self.model(tf.concat([x, y], axis=-1))
                    u = u_v_p[..., 0, tf.newaxis]
                    v = u_v_p[..., 1, tf.newaxis]
                    p = u_v_p[..., 2, tf.newaxis]
                u_x = g.batch_jacobian(u, x)[..., 0]
                v_x = g.batch_jacobian(v, x)[..., 0]
                u_y = g.batch_jacobian(u, y)[..., 0]
                v_y = g.batch_jacobian(v, y)[..., 0]
                p_x = g.batch_jacobian(p, x)[..., 0]
                p_y = g.batch_jacobian(p, y)[..., 0]

                del g
            u_xx = gg.batch_jacobian(u_x, x)[..., 0]
            u_yy = gg.batch_jacobian(u_y, y)[..., 0]

            v_xx = gg.batch_jacobian(v_x, x)[..., 0]
            v_yy = gg.batch_jacobian(v_y, y)[..., 0]

            del gg
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
    Now only takes (x, y) as input and outputs (u, v, p).
    Losses for PDE, BC, etc., should be computed outside this class.
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
            u, v, p: numpy arrays of shape (N, 1)
        """
        preds = self.network(xy)
        u = preds[..., 0:1]
        v = preds[..., 1:2]
        p = preds[..., 2:3]
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



"""mark2_pinn.py"""


def build(self):
        """
        Build a PINN model for the steady Navier-Stokes equation.
        Returns:
            PINN model for the steady Navier-Stokes equation with
                input: [ (x, y) relative to equation,
                         (x, y) relative to boundary condition ],
                output: [ (u, v) relative to equation (must be zero),
                          (psi, psi) relative to boundary condition (psi is duplicated because outputs require the same dimensions),
                          (u, v) relative to boundary condition ]
        """

        # equation input: (x, y)
        xy_eqn = tf.keras.layers.Input(shape=(2,))
        xy_roi_1 = tf.keras.layers.Input(shape=(2,))
        xy_roi_2 = tf.keras.layers.Input(shape=(2,))
        # boundary condition
        xy_in = tf.keras.layers.Input(shape=(2,))
        xy_out = tf.keras.layers.Input(shape=(2,))
        xy_w1 = tf.keras.layers.Input(shape=(2,))
        xy_w2 = tf.keras.layers.Input(shape=(2,))
        xy_circle = tf.keras.layers.Input(shape=(2,))

        # compute gradients relative to equation
        p_grads, u_grads, v_grads = self.grads(xy_eqn)
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads
        # compute equation loss
        u_eqn =  u*u_x + v*u_y + p_x/self.rho - self.mu*(u_xx + u_yy) / self.rho
        v_eqn =  u*v_x + v*v_y + p_y/self.rho - self.mu*(v_xx + v_yy) / self.rho
        uv_eqn = u_x + v_y
        uv_eqn_int = tf.concat([u_eqn, v_eqn, uv_eqn], axis=-1)

        # compute gradients to eqn relative to roi
        p_grads_r1, u_grads_r1, v_grads_r1 = self.grads(xy_roi_1)
        _, p_x, p_y = p_grads_r1
        u, u_x, u_y, u_xx, u_yy = u_grads_r1
        v, v_x, v_y, v_xx, v_yy = v_grads_r1
        # compute equation loss
        u_eqn =  u*u_x + v*u_y + p_x/self.rho - self.mu*(u_xx + u_yy) / self.rho
        v_eqn =  u*v_x + v*v_y + p_y/self.rho - self.mu*(v_xx + v_yy) / self.rho
        uv_eqn = u_x + v_y
        uv_eqn_roi_1 = tf.concat([u_eqn, v_eqn, uv_eqn], axis=-1)

        # compute gradients to eqn relative to roi
        p_grads_r2, u_grads_r2, v_grads_r2 = self.grads(xy_roi_2)
        _, p_x, p_y = p_grads_r2
        u, u_x, u_y, u_xx, u_yy = u_grads_r2
        v, v_x, v_y, v_xx, v_yy = v_grads_r2
        # compute equation loss
        u_eqn =  u*u_x + v*u_y + p_x/self.rho - self.mu*(u_xx + u_yy) / self.rho
        v_eqn =  u*v_x + v*v_y + p_y/self.rho - self.mu*(v_xx + v_yy) / self.rho
        uv_eqn = u_x + v_y
        uv_eqn_roi_2 = tf.concat([u_eqn, v_eqn, uv_eqn], axis=-1)

        # compute gradients relative to boundary condition
        p_r, u_grads_r, v_grads_r = self.grads(xy_out)
        uv_out = tf.concat([p_r[0], p_r[0], p_r[0]], axis=-1)

        p_l, u_grads_l, v_grads_l = self.grads(xy_w1)
        uv_w1 = tf.concat([u_grads_l[0], v_grads_l[0], p_l[2]], axis=-1)
        
        p_l, u_grads_l, v_grads_l = self.grads(xy_w2)
        uv_w2 = tf.concat([u_grads_l[0], v_grads_l[0], p_l[2]], axis=-1)
        
        p_l, u_grads_l, v_grads_l = self.grads(xy_circle)
        uv_circle = tf.concat([u_grads_l[0], v_grads_l[0], u_grads_l[0]], axis=-1)

        p_inn, u_inn, v_inn = self.grads(xy_in)
        uv_in = tf.concat([u_inn[0], v_inn[0], u_inn[0]], axis=-1)

        # build the PINN model for the steady Navier-Stokes equation
        return tf.keras.models.Model(
            inputs=[xy_eqn, xy_roi_1, xy_roi_2, xy_w1, xy_w2, xy_out, xy_in, xy_circle], outputs=[uv_eqn_int, uv_eqn_roi_1, uv_eqn_roi_2, uv_in, uv_out, uv_w1, uv_w2, uv_circle]) # tensor order w.r.t original code



"""optimizer.py"""

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

    def __init__(self, model, x_train, y_train, factr=1e5, m=50, maxls=50, maxiter=30000):
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
        Evaluate loss and gradients for weights as tf.Tensor.
        Args:
            x: input data.
        Returns:
            loss and gradients for weights as tf.Tensor.
        """

        with tf.GradientTape() as g:
            # LOSS FUNCTION IS SPECIFIED HERE:
            loss = tf.reduce_mean(tf.keras.losses.logcosh(self.model(x), y))
            # To use MSE instead, use:
            # loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(self.model(x), y))
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


"""tf_silent.py"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



"""main.py"""


# build a core network model
network = Network().build()
network.summary()
# build a PINN model
pinn = PINN(network, rho=rho, mu=mu).build()

# Domain and circle data
x_f =2
x_ini=0
y_f=1
y_ini=0
Cx = 0.5
Cy = 0.5
a = 0.1
b = 0.1

xyt_circle = np.random.rand(num_train_samples, 2)
xyt_circle[...,0] = 2*(a)*xyt_circle[...,0] +(Cx-a)
xyt_circle[0:num_train_samples//2,1] = b*(1 - (xyt_circle[0:num_train_samples//2,0]-Cx)**2 / a**2)**0.5 + Cy
xyt_circle[num_train_samples//2:,1] = -b*(1 - (xyt_circle[num_train_samples//2:,0]-Cx)**2 / a**2)**0.5 + Cy

# Interior domain sampling
xyt_eqn = np.random.rand(num_train_samples, 2)
xyt_eqn[...,0] = (x_f - x_ini)*xyt_eqn[...,0] + x_ini
xyt_eqn[...,1] = (y_f - y_ini)*xyt_eqn[...,1] + y_ini

# remove points inside the circle
for i in range(num_train_samples):
  while (xyt_eqn[i, 0] - Cx)**2/a**2 + (xyt_eqn[i, 1] - Cy)**2/b**2 < 1:
    xyt_eqn[i, 0] = (x_f - x_ini) * np.random.rand(1, 1) + x_ini
    xyt_eqn[i, 1] = (y_f - y_ini) * np.random.rand(1, 1) + y_ini

# Sample in the annulus (r+0.2) region # annulus is a plane region between two concentric circles
num_annulus_samples = 2000
theta = np.random.uniform(0, 2*np.pi, num_annulus_samples)
r = np.sqrt(np.random.uniform(1, 6.25, num_annulus_samples))  # sqrt for uniform area
# circle eqn in polar coordinates: x = Cx + a * r * cos(theta), y = Cy + b * r * sin(theta)
x = Cx + a * r * np.cos(theta)
y = Cy + b * r * np.sin(theta)
xyt_annulus = np.stack([x, y], axis=1)

# Wake region
num_strip_samples = 4000
x_strip = np.random.uniform(0.55, 0.75, num_strip_samples)
y_strip = np.random.uniform(0.3, 0.7, num_strip_samples)
xyt_strip = np.stack([x_strip, y_strip], axis=1)
for i in range(num_strip_samples):
  while ((xyt_strip[i, 0] - Cx)**2/a**2 + (xyt_strip[i, 1] - Cy)**2/b**2) < 1:
    xyt_strip[i, 0] = np.random.uniform(0.55, 0.75)
    xyt_strip[i, 1] = np.random.uniform(0.3, 0.7)


xyt_roi = np.concatenate([xyt_annulus, xyt_strip], axis=0)
num_interior_samples = xyt_roi.shape[0]

xyt_w1 = np.random.rand(num_train_samples, 2)  # top-bottom boundaries
xyt_w1[..., 0] = (x_f - x_ini)*xyt_w1[...,0] + x_ini
xyt_w1[..., 1] =  y_ini          # y-position is 0 or 1
num_w1_samples = xyt_w1.shape[0]

xyt_w2 = np.random.rand(num_train_samples, 2)  # top-bottom boundaries
xyt_w2[..., 0] = (x_f - x_ini)*xyt_w2[...,0] + x_ini
xyt_w2[..., 1] =  y_f
num_w2_samples = xyt_w2.shape[0]

xyt_out = np.random.rand(num_train_samples, 2)  # left-right boundaries
xyt_out[..., 0] = x_f
num_out_samples = xyt_out.shape[0]

xyt_in = np.random.rand(num_train_samples, 2)
xyt_in[..., 0] = x_ini
num_in_samples = xyt_in.shape[0]

# Aggregate input data for training
x_train = [
    xyt_eqn,        # All interior points (PDE loss)
    xyt_roi,  # Region of interest (annulus + strip)
    xyt_w1,            # Wall y=0
    xyt_w2,            # Wall y=1
    xyt_out,           # Outlet x=2
    xyt_in,            # Inlet x=0
    xyt_circle         # Cylinder boundary
]
# y_train: create outputs with respect to data shapes for each region
zeros_interior = np.zeros((xyt_eqn.shape[0], 3))
zeros_roi = np.zeros((xyt_roi.shape[0], 3))
zeros_w1 = np.zeros((xyt_w1.shape[0], 3))
zeros_w2 = np.zeros((xyt_w2.shape[0], 3))
zeros_out = np.zeros((xyt_out.shape[0], 3))
zeros_circle = np.zeros((xyt_circle.shape[0], 3))

# Inlet boundary condition (update shape)
a = u_0(tf.constant(xyt_in)).numpy()
b = np.zeros((xyt_in.shape[0], 1))
onze = np.random.permutation(np.concatenate([a, b, a], axis=-1))

y_train = [
    zeros_interior,  # All interior points (PDE loss)
    zeros_roi,       # Region of interest (annulus + strip)
    onze,            # Inlet x=0
    zeros_w1,        # Wall y=0
    zeros_w2,        # Wall y=1
    zeros_out,       # Outlet x=2
    zeros_circle     # Cylinder boundary
]



from optimizer import L_BFGS_B
# train the model using L-BFGS-B algorithm
lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train, factr=1e5, m=50, maxls=50, maxiter=30000)
lbfgs.fit()




from optimizer import L_BFGS_B
# train the model using L-BFGS-B algorithm
lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train, factr=1e5, m=50, maxls=50, maxiter=30000)
lbfgs.fit()



