#!/usr/bin/env python3
import collections
import csv
import datetime
import dateutil.parser
import numpy as np
import pickle
from scipy.integrate import odeint

from common import *


# ASSUMPTIONS:
DAYS_INFECTION_TO_DEATH = 17
  # On average, how long does a case of COVID19 last?
AVERAGE_DEATH_RATE = 0.01
  # What fraction of COVID19 cases result in death?
INFECTION_GROWTH_RATE = 0.24
  # How fast does the infection grow if there are very few cases?
  # Assumes no interventions.
INTERVENTION_INFECTION_GROWTH_RATE = {
        'Lockdown': 0.075,
        '~Lockdown': 0.1375,
        'Social distancing': 0.2}
  # Same question when there are active interventions.
DAYS_FORECAST = 60
  # How many days into the future do we simulate?


# Model parameters:

gamma = 1/DAYS_INFECTION_TO_DEATH
  # If we're going to assume the illness takes DAYS_INFECTON_TO_DEATH
  # to run its course, the probability of someone recovering (or dying)
  # needs to be the reciprocal of the duration.

default_beta = np.log(1 + INFECTION_GROWTH_RATE) + gamma
  # The term which grows I in SIR is beta*(S/N)*I - gamma*I
  # When S ~ N (early days of an epidemic), this is just (beta-gamma)*I.
  # So beta-gamma is the infection growth rate.


# The SIR Model:
# See: https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
# Beta is now time-dependant so interventions can turn on and off.
def SIR_deriv(y, t, N, beta, gamma):
    S, I, R = y
    beta_val = beta(t)
    dSdt = -beta_val * S * I / N
    dIdt = beta_val * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Load the data:
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
        if change in INTERVENTION_INFECTION_GROWTH_RATE:
            b = np.log(1 + INTERVENTION_INFECTION_GROWTH_RATE[change]) + gamma
            interventions.append((t,b))
    interventions.sort()
    def beta(t):
        for iv_t, iv_b in interventions:
            if t >= iv_t: return iv_b
        return default_beta
    return beta


# Outputs:
infected_w = csv.writer(open('sir_estimated_infected.csv', 'w'))
infected_w.writerow(
        ["Province/State", "Country/Region", "Lat", "Long",
        "Estimated", "Region Population", "Estimated Per Capita"])


# Start calculating:
for k in sorted(population.keys()):
    if k not in places: continue
    # Compute y0, the initial conditions:
    ts = places[k]
    N = population[k]
    latest_date = ts.dates[-1]
    latest_deaths = ts.deaths[-1]

    start_day = latest_date - datetime.timedelta(DAYS_INFECTION_TO_DEATH)
    start_day_idx = ts.dates.index(start_day)
    start_day_deaths = ts.deaths[start_day_idx]
    start_day_I = (latest_deaths - start_day_deaths) / AVERAGE_DEATH_RATE
    start_day_R = start_day_deaths / AVERAGE_DEATH_RATE
    start_day_S = population[k] - start_day_I - start_day_R

    y0 = start_day_S, start_day_I, start_day_R
    #print("place=" + ','.join(k), " d=" + start_day.isoformat(),
    #        "y0=", y0)

    days_simulation = DAYS_INFECTION_TO_DEATH + DAYS_FORECAST
    t = np.linspace(0, days_simulation, days_simulation)

    beta = interventions_to_beta(
            interventions[(k[0], '', '')] +
            interventions[(k[0], k[1], '')],
            start_day)

    ret = odeint(SIR_deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    estimated = I[DAYS_INFECTION_TO_DEATH]
    row_start = [k[0], k[1], ts.latitude, ts.longitude]
    if estimated > 1000:
        infected_w.writerow(row_start +
            [np.round(estimated, -3), N,
            str(np.round((estimated/N)*100, 2)) + '%'])
    else:
        infected_w.writerow(row_start + ['', N, ''])
