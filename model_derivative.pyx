# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# Disable a warning caused by cython using an old version of numpy.

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef np.ndarray model_derivative(np.ndarray y, float t, float beta, np.ndarray p):
    cdef float S = y[0]
    cdef float E = y[1]
    cdef float I = y[2]
    cdef float H = y[3]
    cdef float D = y[4]
    cdef float R = y[5]
    cdef float exposed_leave_rate = p[0]
    cdef float infectious_leave_rate = p[1]
    cdef float hospital_leave_rate = p[2]
    cdef float hospital_p = p[3]
    cdef float death_p = p[4]
    cdef float N = S+E+I+H+R
    cdef float correction = S/N
    cdef float SE_flow = I*beta*correction
    cdef float EI_flow = E*exposed_leave_rate
    cdef float IH_flow = I*infectious_leave_rate*hospital_p
    cdef float IR_flow = I*infectious_leave_rate*(1-hospital_p)
    cdef float HD_flow = H*hospital_leave_rate*death_p
    cdef float HR_flow = H*hospital_leave_rate*(1-death_p)
    cdef np.ndarray r = np.zeros([6], dtype=DTYPE)
    r[0] = -SE_flow    # dSdt
    r[1] = SE_flow - EI_flow    # dEdt
    r[2] = EI_flow - IH_flow - IR_flow    # dIdt
    r[3] = IH_flow - HD_flow - HR_flow    # dHdt
    r[4] = HD_flow    # dDdt
    r[5] = IR_flow + HR_flow    # dRdt
    return r

def integrate_model(
        np.ndarray[DTYPE_t, ndim=1] y0,
        np.ndarray[DTYPE_t, ndim=1] ts,
        np.ndarray[DTYPE_t, ndim=1] beta_ts,
        np.ndarray[DTYPE_t, ndim=1] beta_vals,
        np.ndarray[DTYPE_t, ndim=1] params):
    cdef float exposed_leave_rate = params[0]
    cdef float infectious_leave_rate = params[1]
    cdef float hospital_leave_rate = params[2]
    cdef float hospital_p = params[3]
    cdef float death_p = params[4]
    cdef float S = y0[0]
    cdef float E = y0[1]
    cdef float I = y0[2]
    cdef float H = y0[3]
    cdef float D = y0[4]
    cdef float R = y0[5]
    cdef float N
    cdef float correction
    cdef float SE_flow
    cdef float EI_flow
    cdef float IH_flow
    cdef float IR_flow
    cdef float HD_flow
    cdef float HR_flow
    cdef float dSdt
    cdef float dEdt
    cdef float dIdt
    cdef float dHdt
    cdef float dDdt
    cdef float dRdt
    cdef int t_idx = 0
    cdef int t_idx_max = len(ts)
    cdef float t = ts[0]
    cdef float step = 0.10
    cdef float beta
    cdef int beta_idx = 0
    cdef int beta_idx_last = len(beta_ts)-1
    cdef float beta_interp_frac
    cdef np.ndarray[DTYPE_t, ndim=2] results = np.zeros((len(ts),6), dtype=DTYPE)
    while True:
        while t >= ts[t_idx]:
            results[t_idx, 0] = S
            results[t_idx, 1] = E
            results[t_idx, 2] = I
            results[t_idx, 3] = H
            results[t_idx, 4] = D
            results[t_idx, 5] = R
            t_idx += 1
            if t_idx >= t_idx_max: break
        if t_idx >= t_idx_max: break

        while beta_idx < beta_idx_last and beta_ts[beta_idx+1] < t: beta_idx += 1

        if beta_idx >= beta_idx_last:
            beta = beta_vals[beta_idx_last]
        else:
            beta_interp_frac = (beta_ts[beta_idx+1] - t)/(beta_ts[beta_idx+1] - beta_ts[beta_idx])
            beta = beta_interp_frac * beta_vals[beta_idx+1] + (1 - beta_interp_frac) * beta_vals[beta_idx]

        N = S+E+I+H+R
        correction = S/N
        SE_flow = I*beta*correction
        EI_flow = E*exposed_leave_rate
        IH_flow = I*infectious_leave_rate*hospital_p
        IR_flow = I*infectious_leave_rate*(1-hospital_p)
        HD_flow = H*hospital_leave_rate*death_p
        HR_flow = H*hospital_leave_rate*(1-death_p)
        dSdt = -SE_flow
        dEdt = SE_flow - EI_flow
        dIdt = EI_flow - IH_flow - IR_flow
        dHdt = IH_flow - HD_flow - HR_flow
        dDdt = HD_flow
        dRdt = IR_flow + HR_flow
        S += dSdt*step
        E += dEdt*step
        I += dIdt*step
        H += dHdt*step
        D += dDdt*step
        R += dRdt*step
        t += step
    return results

