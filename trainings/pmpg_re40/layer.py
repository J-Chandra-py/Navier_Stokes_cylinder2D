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
