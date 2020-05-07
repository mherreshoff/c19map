#!/usr/bin/env python3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pymc3 as pm
import theano
import theano.tensor as tt

from augmented_seir_op import AugmentedSeir

death_obs_val = np.array([
    1,  1,  2,  4,  4,  4,  4,  4,  4,  4,  6,  6,  7,  7, 11, 14, 14, 15, 15, 18, 18, 21, 23, 25, 30, 30, 30, 33, 36, 36, 40,
    42, 43, 47, 50, 57, 58, 60, 64, 68, 72])
population = 36643815

ode_model = AugmentedSeir([0], np.arange(len(death_obs_val)).astype(float), 0.1)

with pm.Model() as model:
    PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
    ProbNormal = pm.Bound(pm.Normal, lower=0.0, upper=1)
    contact_rate = pm.Uniform('contact_rate', 0, 1)
    exposed_leave_rate = PositiveNormal('exposed_leave_rate', mu=1/3.5, sigma=.01)
    infectious_leave_rate = PositiveNormal('infectious_leave_rate', mu=1/4, sigma=0.01)
    hospital_leave_rate = PositiveNormal('hospital_leave_rate', mu=1/9.75, sigma=0.01)
    hospital_p = ProbNormal('hospital_p', mu=0.0714, sigma=0.01)
    death_p = ProbNormal('death_p', mu=0.14, sigma=0.01)
    obs_sigma = PositiveNormal('obs_sigma', mu=10, sigma=1)

    met = pm.Metropolis([
        contact_rate,
        exposed_leave_rate,
        infectious_leave_rate,
        hospital_leave_rate,
        hospital_p,
        death_p,
        obs_sigma])

    y0 = tt.as_tensor_variable(np.array([population, 1, 0,0,0,0],dtype=float))
    params = tt.as_tensor_variable([
        exposed_leave_rate,
        infectious_leave_rate,
        hospital_leave_rate,
        hospital_p,
        death_p])
    betas = tt.as_tensor_variable([contact_rate])
    ode_soln = ode_model(y0, params, betas)

    deaths_soln = ode_soln[:, 4]
    deaths_obs = pm.Normal('deaths_obs', mu=deaths_soln, sd=obs_sigma, observed=death_obs_val)

    #model.profile(model.logpt).summary()
    #model.profile(pm.gradient(model.logpt, model.vars)).summary()

    prior = pm.sample_prior_predictive()
    trace = pm.sample(20000, cores=4, step=met)
    posterior_predictive = pm.sample_posterior_predictive(trace)

    data = az.from_pymc3(
            trace=trace, prior=prior,
            posterior_predictive=posterior_predictive)
    az.plot_posterior(data)
    plt.show()
