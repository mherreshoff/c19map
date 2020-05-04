#!/usr/bin/env python3
import arviz as az
import numpy as np
import pickle
import pymc3 as pm
import theano
import theano.tensor as tt

model_variables = "SEIHDR"
var_to_id = {v: i for i, v in enumerate(model_variables)}

def model_derivative(y, t, p):
    print(type(y), type(t), type(p))
    S = y[0]
    E = y[1]
    I = y[2]
    H = y[3]
    D = y[4]
    R = y[5]
    contact_rate = p[0]
    exposed_leave_rate = p[1]
    infectious_leave_rate = p[2]
    hospital_leave_rate = p[3]
    hospital_p = p[4]
    death_p = p[5]
    N = S+E+I+H+R
    correction = S/N
    SE_flow = I*contact_rate*correction
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
    return [dSdt, dEdt, dIdt, dHdt, dDdt, dRdt]

death_obs_val = np.array([
    1,  1,  2,  4,  4,  4,  4,  4,  4,  4,  6,  6,  7,  7, 11, 14, 14, 15, 15, 18, 18, 21, 23, 25, 30, 30, 30, 33, 36, 36, 40,
    42, 43, 47, 50, 57, 58, 60, 64, 68, 72])
population = 36643815

ode_model = pm.ode.DifferentialEquation(
        func=model_derivative,
        times=np.arange(0,len(death_obs_val)),
        n_states=6,
        n_theta=6)


with pm.Model() as model:
    contact_rate = pm.Uniform('contact_rate', 0, 1)
        # TODO: make time dependant.
    exposed_leave_rate = pm.Uniform('exposed_leave_rate', 0.1, 1)
    infectious_leave_rate = pm.Uniform('infectious_leave_rate', 0.1, 1)
    hospital_leave_rate = pm.Uniform('hospital_leave_rate', 0.05, 1)
    hospital_p = pm.Uniform('hospital_p', 0, 1)
    death_p = pm.Uniform('death_p', 0, 1)

    ode_soln = ode_model(y0=[population, 1, 0,0,0,0], theta = [
        contact_rate,
        exposed_leave_rate,
        infectious_leave_rate,
        hospital_leave_rate,
        hospital_p,
        death_p])

    obs_sigma = pm.HalfCauchy('sigma', 100)
    deaths_soln = ode_soln[:, 4]
    deaths_obs = pm.Normal('deaths_obs', mu=deaths_soln, sd=obs_sigma, observed=death_obs_val)

    #model.profile(model.logpt).summary()
    #model.profile(pm.gradient(model.logpt, model.vars)).summary()

    prior = pm.sample_prior_predictive()
    trace = pm.sample(20, cores=4)
    posterior_predictive = pm.sample_posterior_predictive(trace)

    data = az.from_pymc3(
            trace=trace, prior=prior,
            posterior_predictive=posterior_predictive)
    az.plot_posterior(data)
