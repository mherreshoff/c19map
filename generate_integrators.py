#!/usr/bin/env python3
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

import simple_templates

cython_output = open('integrators.pyx', 'w')

cython_header = """# WARNING: This file is auto-generated.  Do not edit.
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# (Disables a warning caused by cython using an old version of numpy.)
from libc.math cimport sin, cos
cimport numpy as np

import numpy as np
import theano
import theano.tensor as tt

DTYPE = np.float
ctypedef np.float_t DTYPE_t
floatX = theano.config.floatX

"""
cython_output.write(cython_header)


# Note: an 'inflence' is a time-dependent parameter with values determined by linear interpolation.
# Time-dependent parameters are changing external factors influencing the system.
def ode_to_cython(
        name,
        variables,
        derivatives,
        parameters=None,
        influences=None):
    if parameters is None: parameters = []
    if influences is None: influences = []

    if set(derivatives.keys()) != set(variables):
        raise ValueError("Every variable must have a derivative, and vise versa.")


    symbols = {}
    for s in variables + parameters + influences: symbols[s] = sp.Symbol(s)
    dydt = {k : parse_expr(v, local_dict=symbols) for k,v in derivatives.items()}

    ddy_dydt = [[sp.diff(dydt[v], sp.Symbol(w)) for w in variables] for v in variables]
    ddp_dydt = [[sp.diff(dydt[v], sp.Symbol(p)) for p in parameters] for v in variables]
    ddi_dydt = [[sp.diff(dydt[v], sp.Symbol(i)) for i in influences] for v in variables]

    num_vars = len(variables)
    num_params = len(parameters)
    num_vp = len(variables) + len(parameters)

    def add(a):
        a = list(a)
        if len(a) == 0: return "0"
        return ' + '.join(a)

    num_interpolation_points = add(f"len({i}_vals)" for i in influences)

    def interpolation_points_before(i):
        terms = []
        for j in influences:
            if i == j: break
            terms.append(f"len({j}_vals)")
        return add(terms)

    influences_vals_vars = "".join(f"{i}_vals, " for i in influences)
    all_influences_vars = "".join(f"self.{i}_ts, {i}_vals, " for i in influences)
    influences_len_list = ",".join(f"len({i}_ts)" for i in influences)

    argument_list = [
        "np.ndarray[DTYPE_t, ndim=1] y0",
        "np.ndarray[DTYPE_t, ndim=1] params"]
    for i in influences:
        argument_list += [
            f"np.ndarray[DTYPE_t, ndim=1] {i}_ts",
            f"np.ndarray[DTYPE_t, ndim=1] {i}_vals"]
    argument_list += [
        "np.ndarray[DTYPE_t, ndim=1] ts",
        "float step"]
    arguments = "\n"+ " "*8 + (",\n" + " "*8).join(argument_list)
    integrator_code = simple_templates.expand("""
# {'='*75}
# Integrators for {name}

{name}_variables = {repr(variables)}
{name}_parameters = {repr(parameters)}
{name}_influences = {repr(influences)}
{name}_derivatives = {repr(derivatives)}
{name}_num_variables = len({name}_variables)
{name}_num_parameters = len({name}_parameters)
{name}_num_influences = len({name}_influences)
%for include_sensitivity in [False, True]


def {name}_integrate{"_with_sensitivity" if include_sensitivity else ""}(
        %for a in argument_list
        {a},
        %end
        ):
    %for i,v in enumerate(variables)
    cdef float {v} = y0[{i}]
    %end
    %for i,p in enumerate(parameters)
    cdef float {p} = params[{i}]
    %end
    %for i in influences
    cdef int {i}_offset = {num_vp}+{interpolation_points_before(i)}
    cdef int {i}_idx = -1
    cdef int {i}_idx_max = len({i}_ts)-1
    cdef float {i}_frac
    cdef float {i}
    %end
    %for v in variables
    cdef float ddt_{v}
    %end
    cdef int t_idx = 0
    cdef int t_idx_max = len(ts)-1
    cdef float t = ts[0]
    cdef int param_count = {num_vp}+{num_interpolation_points}
    cdef np.ndarray[DTYPE_t, ndim=2] trajectory = np.zeros((len(ts),{num_vars}), dtype=DTYPE)
%if include_sensitivity
    cdef np.ndarray[DTYPE_t, ndim=2] dydp = np.zeros(({num_vars}, param_count), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] ddt_dydp = np.zeros(({num_vars}, param_count), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] sensitivity = np.zeros(
        (len(ts),{num_vars},param_count), dtype=DTYPE)
    cdef float ddy_dydt_val
    cdef float ddi_dydt_val

    for v in range({num_vars}):
        dydp[v,v] = 1
%end

    while True:
        while t >= ts[t_idx]:
            %for i,v in enumerate(variables)
            trajectory[t_idx, {i}] = {v}
            %end
            %if include_sensitivity
            for v in range({num_vars}):
                for p in range(param_count):
                    sensitivity[t_idx,v,p] = dydp[v,p]
            %end
            t_idx += 1
            if t_idx > t_idx_max: break
        if t_idx > t_idx_max: break
        %for i in influences
        while {i}_idx < {i}_idx_max and t > {i}_ts[{i}_idx+1]:
            {i}_idx += 1
        if {i}_idx == -1:
            {i} = {i}_vals[0]
        elif {i}_idx == {i}_idx_max:
            {i} = {i}_vals[{i}_idx_max]
        else:
            {i}_frac = (t - {i}_ts[{i}_idx])/({i}_ts[{i}_idx+1] - {i}_ts[{i}_idx])
            {i} = {i}_frac*{i}_vals[{i}_idx+1] + (1-{i}_frac)*{i}_vals[{i}_idx]
        %end
        %for v in variables
        ddt_{v} = {dydt[v]}
        %end
%if include_sensitivity
        # Calculate ddt_dydp:
        # Initialize:
        for v in range({num_vars}):
            for p in range(param_count):
                ddt_dydp[v,p] = 0
        %for v,v_s in enumerate(variables)

        # ddt_dydp: Calculations for {v_s}
        #  - Paths through previous time step variables:
        %   for w,w_s in enumerate(variables)
        %       if ddy_dydt[v][w] != 0
        ddy_dydt_val = {ddy_dydt[v][w]}  # d^2({v_s})/(dt*d({w_s}))
        for p in range(param_count): ddt_dydp[{v},p] += ddy_dydt_val*dydp[{w},p]
        %       end
        %   end
        #  - Paths through fixed parameters:
        %   for p,p_s in enumerate(parameters)
        %       if ddp_dydt[v][p] != 0
        ddt_dydp[{v},{num_vars+p}] += {ddp_dydt[v][p]}  # d^2({v_s})/(dt*d({p_s}))
        %       end
        %   end
        #  - Paths through interpolated paramters:
        %   for i,i_s in enumerate(influences)
        %       if ddi_dydt[v][i] != 0
        ddi_dydp_val = {ddi_dydt[v][i]}  # d^2({v_s})/(dt*d({i_s}))
        if {i_s}_idx == -1:
            ddt_dydp[{v},{i_s}_offset] += ddi_dydp_val
        elif {i_s}_idx == {i_s}_idx_max:
            ddt_dydp[{v},{i_s}_offset+{i_s}_idx_max] += ddi_dydp_val
        else:
            ddt_dydp[{v},{i_s}_offset+{i_s}_idx] += ddi_dydp_val*(1-{i_s}_frac)
            ddt_dydp[{v},{i_s}_offset+{i_s}_idx+1] += ddi_dydp_val*{i_s}_frac
        %       end
        %   end
        %end
%end
        %for v in variables
        {v} += step*ddt_{v}
        %end
%if include_sensitivity
        for v in range({num_vars}):
            for p in range(param_count):
                dydp[v,p] += step * ddt_dydp[v,p]
%end
        t += step
%if include_sensitivity
    return (trajectory, sensitivity)
%end
%if not include_sensitivity
    return trajectory
%end
%end

# {'-'*60}

class {name}_theano_op(tt.Op):
    \"\"\"
    Run the {name} ODE in the theano graph.

    Parameters
    ----------
    %for i in influences
    {i}_ts: array
        Array of times at which the {i} parameter will be specified.
        Must be ascending. ({i} is linearly interpolated between these times.)
    %end
    ts : array
        Array of times at which to evaluate the solution.
        Must be ascending. The first one will be taken as the initial time.
    step : float
        The step size for Euler's algorithm.
    \"\"\"
    itypes = [
            tt.TensorType(floatX, (False,)),  # y0, float vector
            tt.TensorType(floatX, (False,)),  # params, float vector
            %for i in influences
            tt.TensorType(floatX, (False,)),  # {i}_vals, float vector
            %end
    ]
    otypes = [
            tt.TensorType(floatX, (False, False)),  # trajectory: shape (T, S)
    ]
    def __init__(
            self,
            %for i in influences
            {i}_ts,
            %end
            ts,
            step):
        %for i in influences
        self.{i}_ts = np.array({i}_ts, dtype=float)
        %end
        self.ts = np.array(ts, dtype=float)
        self.step = step
        self.n_times = len(ts)
        self.n_variables = {name}_num_variables
        self.n_params = {name}_num_parameters
        param_chunks = [self.n_variables, self.n_params, {influences_len_list}]
        self.chunk_boundaries = np.cumsum(param_chunks[:-1])

    def perform(self, node, inputs, outputs):
        y0, params, {influences_vals_vars} = inputs
        outputs[0][0] = {name}_integrate(
                y0, params, {all_influences_vars} self.ts, self.step)

    def grad(self, inputs, g):
        y0, params, {influences_vals_vars} = inputs
        output, sensitivity = {name}_integrate_with_sensitivity(
                y0, params, {all_influences_vars} self.ts, self.step)
        inputs_g = np.tensordot(g, sensitivity, axes=([0,1], [0,1]))
        return np.split(inputs_g, self.chunk_boundaries)

    def infer_shape(self, node, input_shapes):
        return [(self.n_times, self.n_variables)]
""", globals(), locals())
    cython_output.write(integrator_code)



ode_to_cython(
        name="pendulum",
        variables=["theta", "omega"],
        parameters = ["b", "c"],
        derivatives={
            "theta": "omega",
            "omega": "-b*omega - c*sin(theta)"})

ode_to_cython(
        name="sir",
        variables=["S", "I", "R"],
        parameters=["gamma"],
        influences=["beta"],
        derivatives={
            "S": "-(S*I/(S+I+R))*beta",
            "I": "(S*I/(S+I+R))*beta - gamma * I",
            "R": "gamma * I"})

ode_to_cython(
        name="augmented_seir",
        variables=["S", "E", "I", "H", "D", "R"],
        parameters=[
            "exposed_leave_rate",
            "infectious_leave_rate",
            "hospital_leave_rate",
            "hospital_p",
            "death_p"],
        influences=["beta"],
        derivatives={
            "S": "-I*beta*S/(S+E+I+H+R)",
            "E": "I*beta*S/(S+E+I+H+R) - E*exposed_leave_rate",
            "I": "E*exposed_leave_rate - I*infectious_leave_rate",
            "H": "I*infectious_leave_rate*hospital_p - H*hospital_leave_rate",
            "D": "H*hospital_leave_rate*death_p",
            "R": "I*infectious_leave_rate*(1-hospital_p) + H*hospital_leave_rate*(1-death_p)"
            })

cython_output.close()
