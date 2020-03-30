#!/usr/bin/env python3
import collections
import csv
import datetime
import dateutil.parser
import numpy as np
import pickle
from scipy.integrate import odeint
import sympy as sp
import sys

from common import *


# ASSUMPTIONS:

#TODO: these numbers are copied from neher.  Vet them more thoroughly.
LATENT_PERIOD = 3.5
INFECTIOUS_PERIOD = 4
P_SEVERE = 0.10
P_CRITICAL = 0.30
P_FATAL = 0.35
HOSPITAL_DURATION = 4
ICU_DURATION = 14


DAYS_INFECTION_TO_DEATH = int(LATENT_PERIOD + INFECTIOUS_PERIOD +
        HOSPITAL_DURATION + ICU_DURATION)
  # On average, how long does a case of COVID19 last?

AVERAGE_DEATH_RATE = P_SEVERE*P_CRITICAL*P_FATAL
  # What fraction of COVID19 cases result in death?

INTERVENTION_INFECTION_GROWTH_RATE = {
        'Default': 0.24,
        'Lockdown': 0.075,
        '~Lockdown': 0.1375,
        'Social distancing': 0.2}
  # Same question when there are active interventions.
DAYS_FORECAST = 60
  # How many days into the future do we simulate?

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

interventions = collections.defaultdict(list)

for country, region, change, date, explanation in csv_as_matrix('data_interventions.csv'):
    p = canonicalize_place((country, region, ''))
    if p is None: continue
    interventions[p].append((change, date))

# TODO: switch to using 'Interventions' spreadsheet CSV when ready.
def interventions_to_beta(raw_interventions, start_day):
    # Takes a list of interventions.
    # Returns a function that computes beta from t.
    interventions = []
    for change, date_s in raw_interventions:
        if date_s == '': continue
        try:
            date = dateutil.parser.parse(date_s).date()
        except Exception:
            print("Non-parsing date (" + date_s + ")")
            continue
        t = (date-start_day).days
        if change in INTERVENTION_BETA:
            b = INTERVENTION_BETA[change]
            interventions.append((t,b))
    interventions.sort()
    interventions.reverse()
    def beta(t):
        for iv_t, iv_b in interventions:
            if t >= iv_t: return iv_b
        return INTERVENTION_BETA['Default']
    return beta


# Set up a default version of the model:
model = Model(
        lambda t: INTERVENTION_BETA['Default'],
        LATENT_PERIOD,
        INFECTIOUS_PERIOD,
        P_SEVERE, HOSPITAL_DURATION,
        P_CRITICAL, ICU_DURATION,
        P_FATAL)

# Find the equilibrium state:
m = model.companion_matrix()
m = m[1:,1:]
    # get rid of the 'S' variable.  Equilibrium only makes sense if we're
    # assuming an infinite population to expand into.
w, v = np.linalg.eig(m)
max_eig_id = int(np.argmax(w))
equilibrium_growth_rate = np.exp(w[max_eig_id])
equilibrium_state = v[:,max_eig_id]
equilibrium_state = np.concatenate([[0], equilibrium_state])  # Add back S row.
equilibrium_state /= np.sum(equilibrium_state[1:5]) # Noralize by active cases.


# Outputs:
infected_w = csv.writer(open('compartmental_estimated_infected.csv', 'w'))
infected_w.writerow(
        ["Province/State", "Country/Region", "Lat", "Long",
        "Estimated", "Region Population", "Estimated Per Capita"])

validation_w = csv.writer(open('compartmental_validated.csv', 'w'))
validation_w.writerow(
        ["Province/State", "Country/Region", 
        "Deaths Predicted", "Deaths Actual"])


# Start calculating:
for k in sorted(population.keys()):
    if k not in places: continue
    print("Place=",k)
    # Compute y0, the initial conditions:
    ts = places[k]
    N = population[k]
    latest_date = ts.dates[-1]
    latest_D = ts.deaths[-1]

    day0 = latest_date - datetime.timedelta(DAYS_INFECTION_TO_DEATH)
    day0_idx = ts.dates.index(day0)
    day0_D = ts.deaths[day0_idx]

    infected = (latest_D - day0_D) / AVERAGE_DEATH_RATE
    y0 = equilibrium_state * infected
    y0[0] = N - np.sum(y0)

    print("    ", day0_D, " <- Day 0 deaths.")
    print("    ", y0[5], " <- Day 0 deaths, extrapolated")

    days_simulation = DAYS_INFECTION_TO_DEATH + DAYS_FORECAST + 1
    t = np.linspace(0, days_simulation, days_simulation)

    beta = interventions_to_beta(
            interventions[(k[0], '', '')] +
            interventions[(k[0], k[1], '')],
            day0)
    model.contact_rate = beta

    ret = odeint(lambda *a: model.derivative(*a), y0, t)
    S, E, I, H, C, D, R = ret.T

    estimated = I[DAYS_INFECTION_TO_DEATH]
    deaths_predicted = D[DAYS_INFECTION_TO_DEATH]
    row_start = [k[0], k[1], ts.latitude, ts.longitude]
    if estimated > 1000:
        infected_w.writerow(row_start +
            [np.round(estimated, -3), N,
            str(np.round((estimated/N)*100, 2)) + '%'])
    else:
        infected_w.writerow(row_start + ['', N, ''])
    validation_w.writerow([k[0], k[1], deaths_predicted, latest_D])
