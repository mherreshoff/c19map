#!/usr/bin/env python3
import csv
import datetime
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
DAYS_FORECAST = 60
  # How many days into the future do we simulate?

# Model parameters:
beta = INFECTION_GROWTH_RATE
  # The term which grows I in SIR is beta*(S/N)*I.
  # When S ~ N (early days of an epidemic), this is just beta*I.
  # So beta can just be the infection growth rate.

gamma = 1/DAYS_INFECTION_TO_DEATH
  # If we're going to assume the illness takes DAYS_INFECTON_TO_DEATH
  # to run its course, the probability of someone recovering (or dying)
  # needs to be the reciprocal of the duration.


# The SIR Model:
# See: https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
def SIR_deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Load the data:
places = pickle.load(open('time_series.pkl', 'rb'))

def csv_as_matrix(path):
    return [r for r in csv.reader(open(path, 'r'))][1:]

def parse_int(s):
    return int(s.replace(',', ''))

population = {(r[0], r[1], '') : parse_int(r[3])
        for r in csv_as_matrix('data_population.csv')}


# Start calculating:
for k in sorted(population.keys()):
    if k not in places: continue
    # Compute y0, the initial conditions:
    ts = places[k]
    N = population[k]
    latest_date = ts.dates[-1]
    latest_deaths = ts.deaths[-1]
    start_day = latest_date - datetime.timedelta(DAYS_INFECTION_TO_DEATH)
    start_day_I = latest_deaths / AVERAGE_DEATH_RATE
    start_day_idx = ts.dates.index(start_day)
    start_day_deaths = ts.deaths[start_day_idx]
    start_day_R = start_day_deaths / AVERAGE_DEATH_RATE
    start_day_S = population[k] - start_day_I - start_day_R

    y0 = start_day_S, start_day_I, start_day_R
    #print("place=" + ','.join(k), " d=" + start_day.isoformat(),
    #        "y0=", y0)

    days_simulation = DAYS_INFECTION_TO_DEATH + DAYS_FORECAST
    t = np.linspace(0, days_simulation, days_simulation)
    ret = odeint(SIR_deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    infections_now = I[DAYS_INFECTION_TO_DEATH]
    print(','.join(k),": ", infections_now)
