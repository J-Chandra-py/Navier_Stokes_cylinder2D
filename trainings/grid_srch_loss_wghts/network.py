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
        self._omega = 30.0  # frequency for SIREN activation
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
    
    def siren(self, x):
        """
        SIREN activation function.
        Args:
            x: activation input.
        Returns:
            SIREN output.
        """
        return tf.math.sin(self._omega * x)

    def get_initializer(self, activation):
        """
        Return appropriate weight initializer based on activation function.

        - For 'siren': Use small uniform initializer as recommended in SIREN paper.
        - For others like 'tanh', 'swish', 'mish': Use 'he_normal' to help maintain gradient flow.

        Args:
            activation: string name of the activation function.

        Returns:
            TensorFlow initializer.
        """
        if activation == 'siren':
            # Small weights help sine stay stable in early training
            return tf.keras.initializers.RandomUniform(minval=-1 / self._omega, maxval=1 / self._omega)
        else:
            # he_normal initializer draws weights from a scaled normal distribution to maintain signal variance across layers, especially effective for ReLU-like activations tanh, mish, swish.
            return 'he_normal'

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
                kernel_initializer=self.get_initializer(activation))(x)
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='he_normal')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
