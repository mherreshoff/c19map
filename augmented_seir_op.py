import theano
import theano.tensor as tt

import model_derivative as md

floatX = theano.config.floatX

class AugmentedSeir(tt.Op):
    """
    Run the augmented seir ODE in the theano graph.

    Parameters
    ----------
    ts : array
        Array of times at which to evaluate the solution.
        Must be ascending. The first one will be taken as the initial time.
    beta_ts: array
        Array of times at which the beta parameter will be specified.
        (beta is interpolated between these times.)
    """
    itypes = [
            tt.TensorType(floatX, (False,)),  # y0, float vector
            tt.TensorType(floatX, (False,)),  # params, float vector
            tt.TensorType(floatX, (False,))   # beta_vals, float vector
    ]
    otypes = [
            tt.TensorType(floatX, (False, False)),  # trajectory: shape (T, S)
    ]
    def __init__(self, beta_ts, ts, step):
        self.beta_ts = np.array(beta_ts, dtype=float)
        self.ts = np.array(ts, dtype=float)
        self.step = step
        self.n_times = len(ts)
        self.n_states = md.augmented_seir_num_states
        self.n_fixed_params = md.augmented_seir_num_fixed_params

    def perform(self, node, inputs, outputs):
        y0, params, beta_vals = inputs
        outputs[0][0] = md.integrate_augmented_seir(
                y0, params, beta_ts, beta_vals, ts, step)

    def grad(self, inputs, g):
        y0, params, beta_vals = inputs
        output, sensitivity = md.integrate_augmented_seir_with_sensitivity(
                y0, params, beta_ts, beta_vals, ts, step)
        in_g = np.tensordot(g, sensitivity, axes=([0,1], [0,1]))
        return np.split(in_g, [self.n_states, self.n_states+self.n_fixed_params])

    def infer_shape(self, node, input_shapes):
        return [(self.n_times, self.n_states)]


