#!/usr/bin/env python3

import sympy

import simple_templates


def ode_compile(
        ode_name,
        ode_variables,
        ode_derivatives,
        ode_fixed_parameters,
        ode_interpolated_parameters=None,
        output_file=None):
    if ode_interpolated_parameters is None: ode_interpolated_parameters = []

    num_states = len(ode_variables)

    argument_list = [
        "np.ndarray[DTYPE_t, ndim=1] params"]
    for p in ode_interpolated_parameters:
        argument_list += [
            "np.ndarray[DTYPE_t, ndim=1] {p}_ts",
            "np.ndarray[DTYPE_t, ndim=1] {p}_vals"]
    argument_list += [
        "np.ndarray[DTYPE_t, ndim=1] y0",
        "np.ndarray[DTYPE_t, ndim=1] ts",
        "float step"]
    arguments = "\n"+ " "*8 + (",\n" + " "*8).join(argument_list)


    y_decls = "\n    ".join([f"cdef float {v} = y[{i}]" for i,v in enumerate(ode_variables)])
    fixed_param_decls = "\n    ".join(
        f"cdef float {v} = params[{i}]" for i,v in enumerate(ode_fixed_parameters))
    deriv_decls = "\n    ".join(
        f"cdef float d_{v}_dt = y[{i}]" for i,v in enumerate(ode_variables))

    results_assignment = ("\n"+" "*12).join(
            f"results[t_idx, {i}] = {v}" for i,v in enumerate(ode_variables))

    deriv_calculation = ("\n" + " "*8).join(
            f"d_{v}_dt = {ode_derivatives[v]}" for v in sorted(ode_variables))

    step_ode_variables = ("\n" + " "*8).join(
            f"{v} += d_{v}_dt * step" for v in ode_variables)

    code = simple_templates.expand(
"""# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# (Disables a warning caused by cython using an old version of numpy.)

# WARNING: This file is auto-generated.  Do not edit.

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def integrate_{ode_name}({arguments}):
%for i,v in enumerate(ode_variables)
    cdef float {v} = params[{i}]
%end
    {fixed_param_decls}
    {deriv_decls}
    cdef int t_idx = 0
    cdef int t_idx_max = len(ts)
    cdef float t = ts[0]
    # TODO: put interpolation calc vars here.
    cdef np.ndarray[DTYPE_t, ndim=2] results = np.zeros((len(ts),len(y0)), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] sensitivities = np.zeros((len(ts),{num_states},len(y0)+len(p)), dtype=DTYPE)
        # TODO: Populate these.

    while True:
        while t >= ts[t_idx]:
            {results_assignment}
            t_idx += 1
            if t_idx >= t_idx_max: break
        if t_idx >= t_idx_max: break
        # TODO set up interpolated variables here.
        {deriv_calculation}
        t += step
        {step_ode_variables}
    return result
""", globals(), locals())
    if output_file is not None:
        open(output_file, 'w').write(code)
    else:
        print(code)



ode_compile(
        ode_name="sir_model",
        ode_variables=["S", "I", "R"],
        ode_fixed_parameters=["beta", "gamma"],
        ode_derivatives={
            "S": "-(S*I/(S+I+R))*beta",
            "I": "(S*I/(S+I+R))*beta - gamma * I",
            "R": "gamma * I"})
#        output_file="sir.pyx")

# UNDER CONSTRUCTION:

#   ode_compile(
#       ode_variables=["S", "E", "I", "H", "D", "R"],
#       ode_fixed_parameters=[
#           "exposed_leave_rate",
#           "infectious_leave_rate",
#           "hospital_leave_rate",
#           "hospital_p",
#           "death_p"],
#       ode_interpolated_parameters=["beta"],
#       ode_direvatives={
#           "S": "-I*beta*S/(S+E+I+H+R)",
#           "E": "I*beta*S/(S+E+I+H+R) - E*exposed_leave_rate",
#           "I": "E*exposed_leave_rate + hospit",
#           # TODO
#           }
#       cython_output="some_ode.pyx")

