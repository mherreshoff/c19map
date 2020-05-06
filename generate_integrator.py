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
    num_fixed_params = len(ode_interpolated_parameters)

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
    code = simple_templates.expand(
"""# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# (Disables a warning caused by cython using an old version of numpy.)

# WARNING: This file is auto-generated.  Do not edit.

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def integrate_{ode_name}({arguments}):
    %for i,p in enumerate(ode_fixed_parameters)
    cdef float {p} = params[{i}]
    %end
    %for p in ode_interpolated_parameters
    cdef int {p}_idx = -1
    cdef int {p}_idx_max = len({p}_ts)
    cdef float {p}_frac
    %end
    %for i,v in enumerate(ode_variables)
    cdef float {v} = y0[{i}]
    %end
    %for i,v in enumerate(ode_variables)
    cdef float ddt_{v}
    %end
    cdef int t_idx = 0
    cdef int t_idx_max = len(ts)
    cdef float t = ts[0]
    cdef np.ndarray[DTYPE_t, ndim=2] results = np.zeros((len(ts),len(y0)), dtype=DTYPE)
    #cdef np.ndarray[DTYPE_t, ndim=3] sensitivity_results = np.zeros(
    #    (len(ts),{num_states},len(y0)+len(p)), dtype=DTYPE)

    while True:
        while t >= ts[t_idx]:
            %for i,v in enumerate(ode_variables)
            results[t_idx, {i}] = {v}
            %end
            t_idx += 1
            if t_idx >= t_idx_max: break
        %for p in ode_interpolated_parameters
        while {p}_idx < {p}_idx_max and {p} > {p}_ts[{p}_idx+1]:
            {p}_idx += 1
        if idx_{p} == -1:
            {p} = {p}_vals[0]
        elif idx_{p} == {p}_idx_max:
            {p} = {p}_vals[{p}_idx_max]
        else:
            {p}_frac = (t - {p}_ts[{p}_idx])/({p}_ts[{p}_idx+1] - {p}_ts[{p}_idx])
            {p} = {p}_frac*{p}_vals[{p}_idx+1] + (1-{p}_frac)*{p}_vals[{p}_idx]
        %end
        if t_idx >= t_idx_max: break
        # TODO set up interpolated variables here.
        %for v in ode_variables
        ddt_{v} = {ode_derivatives[v]}
        %end
        %for v in ode_variables
        {v} += step*ddt_{v}
        %end
        t += step
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

ode_compile(
        ode_name="augmented",
        ode_variables=["S", "E", "I", "H", "D", "R"],
        ode_fixed_parameters=[
            "exposed_leave_rate",
            "infectious_leave_rate",
            "hospital_leave_rate",
            "hospital_p",
            "death_p"],
        ode_interpolated_parameters=["beta"],
        ode_derivatives={
            "S": "-I*beta*S/(S+E+I+H+R)",
            "E": "I*beta*S/(S+E+I+H+R) - E*exposed_leave_rate",
            "I": "E*exposed_leave_rate - I*infectious_leave_rate",
            "H": "I*infectious_leave_rate*hospital_p - H*hospital_leave_rate",
            "D": "H*hospital_leave_rate*death_p",
            "R": "I*infectious_leave_rate(1-hospital_p) + H*hospital_leave_rate*(1-death_p)"
            })

