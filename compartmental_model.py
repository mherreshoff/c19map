#!/usr/bin/env python3
import collections
import csv
import datetime
import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.integrate import odeint
import sympy as sp
import sys

from common import *


# ASSUMPTIONS:

LATENT_PERIOD = 3.5
    # Citation: 3 studies in the Midas Database with fairly close agreement.
INFECTIOUS_PERIOD = 4
P_SEVERE = 0.10
P_CRITICAL = 0.30
P_FATAL = 0.35
HOSPITAL_DURATION = 4
ICU_DURATION = 14

INTERVENTION_INFECTION_GROWTH_RATE = {
        'Unknown': 0.24,
        'No Intervention': 0.24,
        'Lockdown': 0.075,
        '~Lockdown': 0.1375,
        'Social Distancing': 0.2}
  # Same question when there are active interventions.

def seir_beta_to_growth_rate(beta):
    sigma = 1/LATENT_PERIOD
    gamma = 1/INFECTIOUS_PERIOD
    m = np.array([
        [-sigma, beta],
        [sigma, -gamma]])
    # Note: this matrix is the linearized version of the SEIR diffeq
    # where S~N for just E and I.
    w, v = np.linalg.eig(m)
    return np.exp(np.max(w)) - 1

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
    variables = "SEIHCDR"
    var_to_id = {v: i for i, v in enumerate(variables)}
    # The SEIHCDR Model.
    # An extension of the SEIR model.
    # See 'About' tab for: https://neherlab.org/covid19/
    def __init__(self,
            contact_rate, latent_t, infectious_t,
            hospital_p, hospital_t, critical_p, critical_t, death_p):
        self.contact_rate = contact_rate
        self.latent_t = latent_t
        self.infectious_t = infectious_t
        self.hospital_p = hospital_p
        self.hospital_t = hospital_t
        self.critical_p = critical_p
        self.critical_t = critical_t
        self.death_p = death_p

    def companion_matrix(self, t=0):
        # Companion matrix for the linear approximation of the model.
        flows = [  # (from variable, to variable, amount variable, rate)
            ('S','E', 'I', self.contact_rate(t)),  # Assumes S ~ N
            ('E','I', 'E', 1/self.latent_t),
            ('I','H', 'I', (1/self.infectious_t)*self.hospital_p),
            ('I','R', 'I', (1/self.infectious_t)*(1-self.hospital_p)),
            ('H','C', 'H', (1/self.hospital_t)*model.critical_p),
            ('H','R', 'H', (1/self.hospital_t)*(1-self.critical_p)),
            ('C','D', 'C', (1/self.critical_t)*self.death_p),
            ('C','R', 'C', (1/self.critical_t)*(1-self.death_p))]
        nv = len(Model.variables)
        m = np.zeros((nv, nv))
        for sv,tv,av,x in flows:
            si = Model.var_to_id[sv]
            ti = Model.var_to_id[tv]
            ai = Model.var_to_id[av]
            m[si][ai] -= x
            m[ti][ai] += x
        return m

    def equilibrium(self):
        # Find the equilibrium state:
        m = model.companion_matrix()
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
        S,E,I,H,C,D,R = y
        N = np.sum(y) - D
        flows = [  # (from variable, to variable, rate)
            ('S','E', self.contact_rate(t) * I * S / N),
            ('E','I', E/self.latent_t),
            ('I','H', (I/self.infectious_t)*self.hospital_p),
            ('I','R', (I/self.infectious_t)*(1-self.hospital_p)),
            ('H','C', (H/self.hospital_t)*model.critical_p),
            ('H','R', (H/self.hospital_t)*(1-self.critical_p)),
            ('C','D', (C/self.critical_t)*self.death_p),
            ('C','R', (C/self.critical_t)*(1-self.death_p))]
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


def default_beta(t):
    return INTERVENTION_BETA['No Intervention']

def interventions_to_beta(iv_dates, iv_strings, zero_day):
    # Takes a list of interventions.
    # Returns a function that computes beta from t.
    if len(iv_dates) == 0: return default_beta
    iv_offsets = [(d-zero_day).days for d in iv_dates]
    lowest = min(iv_offsets)
    highest = max(iv_offsets)
    t_to_beta = {t: INTERVENTION_BETA[s] for t, s in zip(iv_offsets, iv_strings)}
    def beta(t):
        t = int(np.floor(t))
        if t in t_to_beta:
            return t_to_beta[t]
        if t < lowest: return t_to_beta[lowest]
        return t_to_beta[highest]
    return beta


# Set up a default version of the model:
model = Model(
        default_beta,
        LATENT_PERIOD,
        INFECTIOUS_PERIOD,
        P_SEVERE, HOSPITAL_DURATION,
        P_CRITICAL, ICU_DURATION,
        P_FATAL)


# Outputs:
infected_w = csv.writer(open('compartmental_estimated_infected.csv', 'w'))
infected_w.writerow(
        ["Province/State", "Country/Region", "Lat", "Long",
        "Estimated", "Region Population", "Estimated Per Capita"])

comparison_w = csv.writer(open('compartmental_comparison.csv', 'w'))
comparison_w.writerow(
        ["Province/State", "Country/Region", 
        "Deaths Predicted", "Deaths Actual"])

if not os.path.exists('graphs'): os.makedirs('graphs')
graph_days_forecast = 60
  # How many days into the future do we simulate?


# Run the model forward for each of the places:
for k in sorted(population.keys()):
    if k not in places: continue
    ts = places[k]
    N = population[k]
    place_s = ' - '.join([s for s in k if s != ''])
    print("Place =",place_s)

    # We find a region starting where deaths are recorded and ending where
    # an intervention happens to do our curve fit with.
    for i in range(len(ts.deaths)-1):
        if ts.deaths[i+1]<ts.deaths[i]:
            ts.deaths[i+1] = ts.deaths[i]

    nz_deaths, = np.nonzero(ts.deaths)
    if len(nz_deaths) == 0:
        print("No deaths recorded, skipping: ", place_s)
        continue
    fit_start = nz_deaths[0]

    try:
        iv_idx = ts.intervention_dates.index(ts.dates[fit_start])
        first_iv = ts.interventions[iv_idx]
    except:
        iv_idx = 0
        first_iv = 'No Intervention'

    change_iv_date = None
    for i in range(iv_idx, len(ts.intervention_dates)):
        if ts.interventions[iv_idx] != first_iv:
            change_iv_date = ts.intervention_dates[i]
            break

    if change_iv_date:
        fit_end = (change_iv_date-ts.dates[0]).days + int(LATENT_PERIOD)
    else: fit_end = len(ts.dates)

    if fit_end < 0: fit_end = 0
    if fit_end > len(ts.dates): fit_end = len(ts.dates)

    print(ts.dates[fit_start], ts.dates[fit_end-1])

    # We fit the curve to find the starting value.
    # TODO: also find country-custom betas this way.
    fit_len = fit_end-fit_start
    if fit_len < 1:
        print("Interventions reported on or before first death; skipping", place_s)
        continue

    equilibrium_growth_rate, equilibrium_state = model.equilibrium()
    unintervened_deaths = ts.deaths[int(fit_start):int(fit_end)]
    t = np.linspace(0,fit_len-1, fit_len)
    starting_factor = np.exp(np.polyfit(t,
        np.log(unintervened_deaths)-t*np.log(equilibrium_growth_rate), 0))

    y0 = starting_factor * equilibrium_state
    y0[0] = N - np.sum(y0)
    start_idx = fit_start

    present_date = ts.dates[-1]
    days_to_present = len(ts.dates) - 1 - fit_start

    days_simulation = days_to_present + graph_days_forecast + 1
    t = np.linspace(-days_to_present, graph_days_forecast, days_simulation)

    model.contact_rate = interventions_to_beta(
            ts.intervention_dates, ts.interventions, present_date)

    trajectories, info = odeint(lambda *a: model.derivative(*a), y0, t, full_output=True)
    S, E, I, H, C, D, R = trajectories.T

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
