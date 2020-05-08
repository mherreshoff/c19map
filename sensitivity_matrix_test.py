#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import integrators

for i in range(1, 9):
    plt.subplot(2, 4, i)
    in_idx = (i-1) % 4
    out_idx = (i-1) // 4

    in_var = ([s+"_0" for s in integrators.pendulum_variables] + integrators.pendulum_parameters)[in_idx]
    out_var = integrators.pendulum_variables[out_idx]

    b = 0.25
    c = 5.0

    y0 = np.array([np.pi -0.1, 0.0], dtype=float)
    params = np.array([b, c])
    ts = np.linspace(0,10,101)

    trajectory, sensitivity = integrators.pendulum_integrate_with_sensitivity(y0, params, ts, 0.01)

    dv = 0.01
    if in_idx < 2: y0[in_idx] += dv
    else: params[in_idx - 2] += dv
    trajectory2 = integrators.pendulum_integrate(y0, params, ts, 0.01)

    empircal_derivative = (trajectory2[:,out_idx] - trajectory[:,out_idx]) / dv
    calculated_derivative = sensitivity[:,out_idx,in_idx]

    plt.title(f'd({out_var})/d({in_var})')
    plt.plot(ts, empircal_derivative, label='empirical')
    plt.plot(ts, calculated_derivative, label='calculated')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
plt.show()
