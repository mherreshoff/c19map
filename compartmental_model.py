#!/usr/bin/env python3
import collections
import csv
import datetime
import dateutil.parser
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.integrate import odeint
import scipy
import shutil
import sympy as sp
import sys

from common import *


# Model parameters:
LATENT_PERIOD = 3.5
    # 1/sigma - The average length of time between contracting the disease
    #     and becoming infectious.
    # Citation: 3 studies in the Midas Database with fairly close agreement.
INFECTIOUS_PERIOD = 4
    # 1/gamma - The average length of time a person stays infectious
    # TODO: get citation from Brandon.
P_HOSPITAL = 0.10
    # Probability an infectious case gets hospitalized
HOSPITAL_DURATION = 17
    # Average length of hospital stay.
P_DEATH = 0.35
    # Probability of death given hospitaliation

# Growth rates, used to calculate beta(t):
# TODO: change these to have a leading one, simplifying a lot of other stuff.
INTERVENTION_INFECTION_GROWTH_RATE = {
        'Unknown': 0.24,
        'No Intervention': 0.24,
        'Lockdown': 0.075,
        '~Lockdown': 0.1375,
        'Social Distancing': 0.2}

tuned_countries = set(['China', 'Japan', 'Korea, South'])

@functools.lru_cache(maxsize=10000)
def seir_beta_to_growth_rate(beta):
    sigma = 1/LATENT_PERIOD
    gamma = 1/INFECTIOUS_PERIOD
    m = np.array([
        [-sigma, beta],
        [sigma, -gamma]], dtype=float)
    # Note: this matrix is the linearized version of the SEIR diffeq
    # where S~N for just E and I.
    w, v = np.linalg.eig(m)
    return np.exp(np.max(w)) - 1


@functools.lru_cache(maxsize=10000)
def seir_growth_rate_to_beta(igr):
    sigma = 1/LATENT_PERIOD
    gamma = 1/INFECTIOUS_PERIOD
    beta = sp.Symbol('beta')
    target_eigenval = np.log(1 + igr)
    m = sp.Matrix([
        [-sigma, beta],
        [sigma, -gamma]])
    eigenvals = list(m.eigenvals().keys())
        # These are symbolic expressions in terms of beta.
    solns = []
    for eigenval in eigenvals:
        solns += sp.solvers.solve(eigenval-target_eigenval, beta)
    assert len(solns) == 1
    return solns[0]


INTERVENTION_BETA = {n : seir_growth_rate_to_beta(igr)
        for n, igr in INTERVENTION_INFECTION_GROWTH_RATE.items()} 


class Model:
    variables = "SEIHDR"
    var_to_id = {v: i for i, v in enumerate(variables)}
    # The SEIHCDR Model.
    # An extension of the SEIR model.
    # See 'About' tab for: https://neherlab.org/covid19/
    def __init__(self,
            contact_rate, latent_t, infectious_t,
            hospital_p, hospital_t, death_p):
        self.contact_rate = contact_rate
        self.latent_t = latent_t
        self.infectious_t = infectious_t
        self.hospital_p = hospital_p
        self.hospital_t = hospital_t
        self.death_p = death_p

    def companion_matrix(self, t=0):
        # Companion matrix for the linear approximation of the model.
        flows = [  # (from variable, to variable, amount variable, rate)
            ('S','E', 'I', self.contact_rate(t)),  # Assumes S ~ N
            ('E','I', 'E', 1/self.latent_t),
            ('I','H', 'I', (1/self.infectious_t)*self.hospital_p),
            ('I','R', 'I', (1/self.infectious_t)*(1-self.hospital_p)),
            ('H','D', 'H', (1/self.hospital_t)*model.death_p),
            ('H','R', 'H', (1/self.hospital_t)*(1-self.death_p))]
        nv = len(Model.variables)
        m = np.zeros((nv, nv))
        for sv,tv,av,x in flows:
            si = Model.var_to_id[sv]
            ti = Model.var_to_id[tv]
            ai = Model.var_to_id[av]
            m[si][ai] -= x
            m[ti][ai] += x
        return m

    def equilibrium(self, t=0):
        # Find the equilibrium state:
        m = model.companion_matrix(t)
        m = m[1:,1:]
            # Get rid of the 'S' variable.  Equilibrium only makes sense if we're
            # assuming an infinite population to expand into.
        w, v = np.linalg.eig(m)
        max_eig_id = int(np.argmax(w))
        growth_rate = np.exp(w[max_eig_id])
        state = v[:,max_eig_id]
        state = np.concatenate([[0], state])  # Add back S row.
        state /= state[5] # Normalize by deaths.
        return growth_rate, state

    def derivative(self, y, t):
        S,E,I,H,D,R = y
        N = np.sum(y) - D
        flows = [  # (from variable, to variable, rate)
            ('S','E', self.contact_rate(t) * I * S / N),
            ('E','I', E/self.latent_t),
            ('I','H', (I/self.infectious_t)*self.hospital_p),
            ('I','R', (I/self.infectious_t)*(1-self.hospital_p)),
            ('H','D', (H/self.hospital_t)*model.death_p),
            ('H','R', (H/self.hospital_t)*(1-self.death_p))]
        inbound = np.array([
            sum(x for s,t,x in flows if t==v) for v in Model.variables])
        outbound = np.array([
            sum(x for s,t,x in flows if s==v) for v in Model.variables])
        return list(inbound-outbound)


# Load the JHU time series data:
places = pickle.load(open('time_series.pkl', 'rb'))

def parse_int(s):
    return int(s.replace(',', ''))

population = {(r[0], r[1], '') : parse_int(r[3])
        for r in csv_as_matrix('data_population.csv')}


def interventions_to_beta(iv_dates, iv_strings, zero_day,
        growth_rate_power=None):
    # Takes a list of interventions.
    # Returns a function that computes beta from t.
    if growth_rate_power is None:
        betas = INTERVENTION_BETA
    else:
        betas = {}
        for k, b in INTERVENTION_BETA.items():
            gr = seir_beta_to_growth_rate(b)
            gr = (1 + gr) ** growth_rate_power - 1
            betas[k] = seir_growth_rate_to_beta(gr)

    if len(iv_dates) == 0:
        return lambda t:betas['No Intervention']
    iv_offsets = [(d-zero_day).days for d in iv_dates]
    lowest = min(iv_offsets)
    highest = max(iv_offsets)
    t_to_beta = {t: betas[s]
            for t, s in zip(iv_offsets, iv_strings)}
    def beta(t):
        t = int(np.floor(t))
        if t in t_to_beta:
            return t_to_beta[t]
        if t < lowest: return t_to_beta[lowest]
        return t_to_beta[highest]
    return beta


# Set up a default version of the model:
model = Model(
        None,
        LATENT_PERIOD,
        INFECTIOUS_PERIOD,
        P_HOSPITAL, HOSPITAL_DURATION,
        P_DEATH)


# Outputs:
infected_w = csv.writer(open('compartmental_estimated_infected.csv', 'w'))
infected_w.writerow(
        ["Province/State", "Country/Region", "Lat", "Long",
        "Estimated", "Region Population", "Estimated Per Capita"])

comparison_w = csv.writer(open('compartmental_comparison.csv', 'w'))
comparison_w.writerow(
        ["Province/State", "Country/Region", 
        "Deaths Predicted", "Deaths Actual"])

# TODO: add flag for whether graphs happen.
graph_output_dir = 'graphs'
if os.path.exists(graph_output_dir):
    shutil.rmtree(graph_output_dir)
os.makedirs(graph_output_dir)

graph_days_forecast = 60  #TODO: flag.
  # How many days into the future do we simulate?


# Run the model forward for each of the places:
for k in sorted(places.keys()):
    if k not in population: continue

    ts = places[k]
    present_date = ts.dates[-1]
    N = population[k]
    place_s = ' - '.join([s for s in k if s != ''])
    print("Place =",place_s)

    # We find a region starting where deaths are recorded and ending where
    # an intervention happens to do our curve fit with.

    nz_deaths, = np.nonzero(ts.deaths)
    if len(nz_deaths) == 0:
        print("No deaths recorded, skipping: ", place_s)
        continue
    start_idx = nz_deaths[0]
    fit_length = len(ts.dates)-start_idx
    model.contact_rate = interventions_to_beta(
            ts.intervention_dates, ts.interventions, present_date)
    growth_rate, equilibrium_state = model.equilibrium(t=-(fit_length-1))

    t = np.arange(fit_length) - (fit_length+1)
    target = ts.deaths[start_idx:]
    y0 = equilibrium_state.copy()
    y0[0] = (10**9) - np.sum(y0)

    trajectories = odeint(lambda *a: model.derivative(*a), y0, t)
    S, E, I, H, D, R = trajectories.T
    # TODO: cut off when S < .9*N for accuracy in situations 

    if k[0] in tuned_countries:
        def loss(x): return np.linalg.norm((D**x[0])*x[1]-target)
        gr_pow, state_scale = scipy.optimize.minimize(
                loss, [1,1], bounds=[(.2, 1), (.01, 100)]).x
        model.contact_rate = interventions_to_beta(
                ts.intervention_dates, ts.interventions, present_date,
                growth_rate_power=gr_pow)
        growth_rate, equilibrium_state = model.equilibrium(t=-(fit_length-1))
            # Recompute the equilibrium since we've altered the model.
    else:
        def loss(s): return np.linalg.norm(D*s-target)
        state_scale = scipy.optimize.minimize_scalar(loss, bounds=(.01, 100)).x

    present_date = ts.dates[-1]
    days_to_present = len(ts.dates) - 1 - start_idx
    days_simulation = days_to_present + graph_days_forecast + 1
    t = np.arange(days_simulation) - days_to_present
    
    y0 = state_scale * equilibrium_state
    y0[0] = N - np.sum(y0)

    trajectories = odeint(lambda *a: model.derivative(*a), y0, t)
    S, E, I, H, D, R = trajectories.T

    #for k, v in sorted(info.items()):
    #    print("\t", k, ": ", v)

    # Estimation:
    row_start = [k[0], k[1], ts.latitude, ts.longitude]
    estimated = np.round(I[days_to_present], -3)
    if estimated < 1000: estimated = ''
    infected_w.writerow(row_start + [estimated])

    # Latest deaths comparison:
    deaths_predicted = D[days_to_present]
    deaths_actual = ts.deaths[-1]
    comparison_w.writerow([k[0], k[1], deaths_predicted, deaths_actual])

    # Graphs:
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    intervention_starts = []
    old_s = 'No Intervention'
    for d,s in zip(ts.intervention_dates, ts.interventions):
        if s!=old_s: intervention_starts.append((d, s))
        old_s = s
    intervention_s = ', '.join(
            s+" on "+d.isoformat() for d,s in intervention_starts)
    ax.set_title(place_s + "\n" + intervention_s)
    ax.set_xlabel('Days (0 is '+present_date.isoformat()+')')
    ax.set_ylabel('People (log)')
    for var, curve in zip(Model.variables, trajectories.T):
        ax.semilogy(t, curve, label=var)
    ax.semilogy(t[0:days_to_present+1], ts.deaths[start_idx:],
            's', label='D emp.')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.savefig(os.path.join('graphs', place_s + '.png'))
    plt.close('all') # Reset plot for next time.
