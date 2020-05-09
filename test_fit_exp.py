#!/usr/bin/env python3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pymc3 as pm
import theano
import theano.tensor as tt

import integrators

N = 100
ode_model = integrators.compound_theano_op(np.arange(N).astype(float), 0.1)
x_obs_vals = 1.05 ** np.arange(100) + 5*np.random.normal(size=100)

with pm.Model() as model:
    PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
    x = PositiveNormal('x', mu=0, sigma=5)
    g = pm.Bound(pm.Normal, lower=-.3, upper=.3)('g', mu=0, sigma=0.2)
    obs_sigma = PositiveNormal('obs_sigma', mu=0,sigma=5)

    y0 = tt.as_tensor_variable([x])
    params = tt.as_tensor_variable([g])
    trajectory = ode_model(y0, params)
    x_trajectory = trajectory[:,0]

    x_obs = pm.Normal('x_obs', mu=x_trajectory, sigma=obs_sigma, observed=x_obs_vals)

    prior = pm.sample_prior_predictive()
    trace = pm.sample(100, cores=4)
    posterior_predictive = pm.sample_posterior_predictive(trace)

    data = az.from_pymc3(
            trace=trace, prior=prior,
            posterior_predictive=posterior_predictive)
    az.plot_posterior(data)
    plt.show()
    
