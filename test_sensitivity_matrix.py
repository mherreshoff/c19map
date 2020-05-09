#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import integrators

# Influences not yet supported.
def plot_sensitivities(name, y0, params, ts, step):
    def get(s): return integrators.__dict__[f'{name}_{s}']
    integ = get('integrate')
    integ_with_sens = get('integrate_with_sensitivity')
    variables = get('variables')
    input_names = [v+"_0" for v in variables] + get('parameters')
    output_names = get('variables')

    num_inputs = len(input_names)
    num_outputs = len(output_names)

    for i in range(num_inputs*num_outputs):
        plt.subplot(num_outputs, num_inputs, i+1)
        out_idx, in_idx = divmod(i, num_inputs)

        in_var = input_names[in_idx]
        out_var = output_names[out_idx]

        y0_a = np.array(y0, dtype=float)
        params_a = np.array(params, dtype=float)
        ts_a = np.array(ts, dtype=float)

        trajectory, sensitivity = integ_with_sens(y0_a, params_a, ts_a, step)

        dv = 0.01
        if in_idx < len(variables): y0_a[in_idx] += dv
        else: params_a[in_idx - len(variables)] += dv
        trajectory2 = integ(y0_a, params_a, ts_a, 0.01)

        empircal_derivative = (trajectory2[:,out_idx] - trajectory[:,out_idx]) / dv
        calculated_derivative = sensitivity[:,out_idx,in_idx]

        plt.title(f'd({out_var})/d({in_var})')
        plt.plot(ts, empircal_derivative, label='empirical')
        plt.plot(ts, calculated_derivative, label='calculated')
        plt.legend(loc='best')
        plt.grid()
    plt.show()

plot_sensitivities('compound', [1], [0.05], np.arange(100)/10, 0.01)
plot_sensitivities('pendulum', [np.pi - 0.1, 0], [0.25, 5.0], np.arange(100)/10, 0.01)
