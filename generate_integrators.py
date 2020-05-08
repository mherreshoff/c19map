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


def ode_to_cython(
        ode_name,
        ode_variables,
        ode_derivatives,
        ode_fixed_parameters=None,
        ode_interpolated_parameters=None):
    if ode_fixed_parameters is None: ode_fixed_parameters = []
    if ode_interpolated_parameters is None: ode_interpolated_parameters = []

    symbols = {}
    for v in ode_variables: symbols[v] = sp.Symbol(v)
    for v in ode_fixed_parameters: symbols[v] = sp.Symbol(v)
    for v in ode_interpolated_parameters: symbols[v] = sp.Symbol(v)

    ode_derivatives = {k : parse_expr(v, local_dict=symbols)
            for k,v in ode_derivatives.items()}

    ddy_dydt = [[sp.diff(ode_derivatives[v], sp.Symbol(v2)) for v2 in ode_variables]
        for v in ode_variables]

    ddfp_dydt = [[sp.diff(ode_derivatives[v], sp.Symbol(fp)) for fp in ode_fixed_parameters]
            for v in ode_variables]
    ddip_dydt = [[sp.diff(ode_derivatives[v], sp.Symbol(ip)) for ip in ode_interpolated_parameters]
            for v in ode_variables]

    def add(a):
        a = list(a)
        if len(a) == 0: return "0"
        return ' + '.join(a)

    num_vars = len(ode_variables)
    num_fparams = len(ode_fixed_parameters)
    num_iparams = add(f"len({p}_vals)" for p in ode_interpolated_parameters)
    if num_iparams == '': num_iparams = '0'

    def iparams_before(curr_p):
        terms = []
        for p in ode_interpolated_parameters:
            if p == curr_p: break
            terms.append(f"len({p}_vals)")
        return add(terms)

    vals_ip_vars = "".join(f"{ip}_vals, " for ip in ode_interpolated_parameters)
    all_ip_variables = "".join(f"self.{ip}_ts, {ip}_vals, " for ip in ode_interpolated_parameters)
    ip_ts_lens = ",".join(f"len({ip}_ts)" for ip in ode_interpolated_parameters)

    argument_list = [
        "np.ndarray[DTYPE_t, ndim=1] y0",
        "np.ndarray[DTYPE_t, ndim=1] params"]
    for p in ode_interpolated_parameters:
        argument_list += [
            f"np.ndarray[DTYPE_t, ndim=1] {p}_ts",
            f"np.ndarray[DTYPE_t, ndim=1] {p}_vals"]
    argument_list += [
        "np.ndarray[DTYPE_t, ndim=1] ts",
        "float step"]
    arguments = "\n"+ " "*8 + (",\n" + " "*8).join(argument_list)
    integrator_code = simple_templates.expand("""
# {'='*75}
# Integrators for {ode_name}

{ode_name}_num_states = {num_vars}
{ode_name}_num_fixed_params = {num_fparams}
%for include_sensitivity in [False, True]


def {ode_name}_integrate{"_with_sensitivity" if include_sensitivity else ""}(
        %for a in argument_list
        {a},
        %end
        ):
    %for i,v in enumerate(ode_variables)
    cdef float {v} = y0[{i}]
    %end
    %for i,p in enumerate(ode_fixed_parameters)
    cdef float {p} = params[{i}]
    %end
    %for p in ode_interpolated_parameters
    cdef int {p}_offset = {num_vars+num_fparams}+{iparams_before(p)}
    cdef int {p}_idx = -1
    cdef int {p}_idx_max = len({p}_ts)-1
    cdef float {p}_frac
    cdef float {p}
    %end
    %for i,v in enumerate(ode_variables)
    cdef float ddt_{v}
    %end
    cdef int t_idx = 0
    cdef int t_idx_max = len(ts)-1
    cdef float t = ts[0]
    cdef int param_count = {num_vars+num_fparams}+{num_iparams}
    cdef np.ndarray[DTYPE_t, ndim=2] trajectory = np.zeros((len(ts),{num_vars}), dtype=DTYPE)
%if include_sensitivity
    cdef np.ndarray[DTYPE_t, ndim=2] dydp = np.zeros(
        ({num_vars}, {num_vars+num_fparams}+{num_iparams}), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] ddt_dydp = np.zeros(({num_vars}, param_count), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] sensitivity = np.zeros(
        (len(ts),{num_vars},param_count), dtype=DTYPE)
    cdef float ddy_dydt_val
    cdef float ddip_dydt_val

    for v in range({num_vars}):
        dydp[v,v] = 1
%end

    while True:
        while t >= ts[t_idx]:
            %for i,v in enumerate(ode_variables)
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
        %for p in ode_interpolated_parameters
        while {p}_idx < {p}_idx_max and t > {p}_ts[{p}_idx+1]:
            {p}_idx += 1
        if {p}_idx == -1:
            {p} = {p}_vals[0]
        elif {p}_idx == {p}_idx_max:
            {p} = {p}_vals[{p}_idx_max]
        else:
            {p}_frac = (t - {p}_ts[{p}_idx])/({p}_ts[{p}_idx+1] - {p}_ts[{p}_idx])
            {p} = {p}_frac*{p}_vals[{p}_idx+1] + (1-{p}_frac)*{p}_vals[{p}_idx]
        %end
        %for v in ode_variables
        ddt_{v} = {ode_derivatives[v]}
        %end
%if include_sensitivity
        # Calculate ddt_dydp:
        # Initialize:
        for v in range({num_vars}):
            for p in range(param_count):
                ddt_dydp[v,p] = 0
        %for v in range(num_vars)

        # ddt_dydp: Calculations for {ode_variables[v]}:
        #  - Paths through previous time step variables:
        %   for v2 in range(num_vars)
        %       if ddy_dydt[v][v2] != 0
        ddy_dydt_val = {ddy_dydt[v][v2]}
        for p in range(param_count): ddt_dydp[{v},p] += ddy_dydt_val*dydp[{v2},p]
        %       end
        %   end
        #  - Paths through fixed parameters:
        %   for fp in range(num_fparams)
        %       if ddfp_dydt[v][fp] != 0
        ddt_dydp[{v},{num_vars+fp}] += {ddfp_dydt[v][fp]}
        %       end
        %   end
        #  - Paths through interpolated paramters:
        %   for ip,ip_s in enumerate(ode_interpolated_parameters)
        %       if ddip_dydt[v][ip] != 0
        ddip_dydp_val = {ddip_dydt[v][ip]}
        if {ip_s}_idx == -1:
            ddt_dydp[{v},{ip_s}_offset] += ddip_dydp_val
        elif {ip_s}_idx == {ip_s}_idx_max:
            ddt_dydp[{v},{ip_s}_offset+{ip_s}_idx_max] += ddip_dydp_val
        else:
            ddt_dydp[{v},{ip_s}_offset+{ip_s}_idx] += ddip_dydp_val*(1-{ip_s}_frac)
            ddt_dydp[{v},{ip_s}_offset+{ip_s}_idx+1] += ddip_dydp_val*{ip_s}_frac
        %       end
        %   end
        %end
%end
        %for v in ode_variables
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

class {ode_name}_theano_op(tt.Op):
    \"\"\"
    Run the augmented seir ODE in the theano graph.

    Parameters
    ----------
    %for ip in ode_interpolated_parameters
    {ip}_ts: array
        Array of times at which the {ip} parameter will be specified.
        ({ip} is linearly interpolated between these times.)
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
            %for ip in ode_interpolated_parameters
            tt.TensorType(floatX, (False,)),  # {ip}_vals, float vector
            %end
    ]
    otypes = [
            tt.TensorType(floatX, (False, False)),  # trajectory: shape (T, S)
    ]
    def __init__(
            self,
            %for ip in ode_interpolated_parameters
            {ip}_ts,
            %end
            ts,
            step):
        %for ip in ode_interpolated_parameters
        self.{ip}_ts = np.array({ip}_ts, dtype=float)
        %end
        self.ts = np.array(ts, dtype=float)
        self.step = step
        self.n_times = len(ts)
        self.n_states = {ode_name}_num_states
        self.n_fixed_params = {ode_name}_num_fixed_params
        param_chunks = [self.n_states, self.n_fixed_params, {ip_ts_lens}]
        self.chunk_boundaries = np.cumsum(param_chunks[:-1])


    def perform(self, node, inputs, outputs):
        y0, params, {vals_ip_vars} = inputs
        outputs[0][0] = {ode_name}_integrate(
                y0, params, {all_ip_variables} self.ts, self.step)

    def grad(self, inputs, g):
        y0, params, {vals_ip_vars} = inputs
        output, sensitivity = {ode_name}_integrate_with_sensitivity(
                y0, params, {all_ip_variables} self.ts, self.step)
        inputs_g = np.tensordot(g, sensitivity, axes=([0,1], [0,1]))
        return np.split(inputs_g, self.chunk_boundaries)

    def infer_shape(self, node, input_shapes):
        return [(self.n_times, self.n_states)]
""", globals(), locals())
    cython_output.write(integrator_code)



ode_to_cython(
        ode_name="pendulum",
        ode_variables=["theta", "omega"],
        ode_fixed_parameters = ["b", "c"],
        ode_derivatives={
            "theta": "omega",
            "omega": "-b*omega + c*sin(theta)"})

ode_to_cython(
        ode_name="sir",
        ode_variables=["S", "I", "R"],
        ode_fixed_parameters=["gamma"],
        ode_interpolated_parameters=["beta"],
        ode_derivatives={
            "S": "-(S*I/(S+I+R))*beta",
            "I": "(S*I/(S+I+R))*beta - gamma * I",
            "R": "gamma * I"})

ode_to_cython(
        ode_name="augmented_seir",
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
            "R": "I*infectious_leave_rate*(1-hospital_p) + H*hospital_leave_rate*(1-death_p)"
            })

cython_output.close()
