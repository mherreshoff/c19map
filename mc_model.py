#!/usr/bin/env python3
import pickle
import pymc3 as pm
import arviz as az

from common import *

places = pickle.load(open('time_series.pkl', 'rb'))

model_variables = "SEIHDR"
var_to_id = {v: i for i, v in enumerate(model_variables)}

def model_derivative(y, t, p):
    S = y[0]
    E = y[1]
    I = y[2]
    H = y[3]
    D = y[4]
    R = y[5]
    contact_rate = p[0]
    exposed_t = p[1]
    infectious_t = p[2]
    hospital_t = p[3]
    hospital_p = p[4]
    death_p = p[5]
    N = S+E+I+H+R
    correction = S/N
    flows = [  # (from variable, to variable, amount variable, rate)
        ('S','E', 'I', contact_rate*correction),
        ('E','I', 'E', 1/exposed_t),
        ('I','H', 'I', (1/infectious_t)*hospital_p),
        ('I','R', 'I', (1/infectious_t)*(1-hospital_p)),
        ('H','D', 'H', (1/hospital_t)*death_p),
        ('H','R', 'H', (1/hospital_t)*(1-death_p))]
    nv = len(model_variables)
    r = [0 for n in range(6)]
    for sv,tv,av,x in flows:
        si = var_to_id[sv]
        ti = var_to_id[tv]
        ai = var_to_id[av]
        r[si] -= x*y[ai]
        r[ti] += x*y[ai]
    return r

p = places[('Afghanistan','','')]
nz_deaths_idx, = np.nonzero(p.deaths)
first_death = nz_deaths_idx[0]

ode_model = pm.ode.DifferentialEquation(
        func=model_derivative,
        times=np.arange(0,len(p.deaths)-first_death),
        n_states=6,
        n_theta=6)


with pm.Model() as model:
    contact_rate = pm.Uniform('contact_rate', 0, 1)
        # TODO: make time dependant.
    exposed_t = pm.Uniform('exposed_t', 1, 8)
    infectious_t = pm.Uniform('infectious_t', 1, 8)
    hospitalized_t = pm.Uniform('hospital_t', 1, 20)
    hospitalized_p = pm.Uniform('hospital_p', 0, 1)
    death_p = pm.Uniform('death_p', 0, 1)

    ode_soln = ode_model(y0=[p.population, 1, 0,0,0,0], theta = [
        contact_rate,
        exposed_t,
        infectious_t,
        hospitalized_t,
        hospitalized_p,
        death_p])

    obs_sigma = pm.HalfCauchy('sigma', 100)
    deaths_soln = ode_soln[:, 4]
    deaths_obs = pm.Normal('deaths_obs', mu=deaths_soln, sd=obs_sigma, observed=p.deaths[first_death:])

    prior = pm.sample_prior_predictive()
    trace = pm.sample(20, cores=4)
    posterior_predictive = pm.sample_posterior_predictive(trace)

    data = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=posterior_predictive)
az.plot_posterior(data)
