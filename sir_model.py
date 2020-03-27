#!/usr/bin/env python3
import csv
import datetime
import pickle

from common import *


# Assumptions:
DAYS_INFECTION_TO_DEATH = 17
AVERAGE_DEATH_RATE = 0.01


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
    latest_date = ts.dates[-1]
    latest_deaths = ts.deaths[-1]
    start_day = latest_date - datetime.timedelta(DAYS_INFECTION_TO_DEATH)
    start_day_I = latest_deaths / AVERAGE_DEATH_RATE
    start_day_idx = ts.dates.index(start_day)
    start_day_deaths = ts.deaths[start_day_idx]
    start_day_R = start_day_deaths / AVERAGE_DEATH_RATE
    start_day_S = population[k] - start_day_I - start_day_R

    y0 = start_day_S, start_day_I, start_day_R
    print("place=" + ','.join(k), " d=" + start_day.isoformat(),
            "S=",start_day_S,"I=",start_day_I,"R=",start_day_R)
