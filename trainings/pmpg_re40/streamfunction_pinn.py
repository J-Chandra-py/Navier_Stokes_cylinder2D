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
