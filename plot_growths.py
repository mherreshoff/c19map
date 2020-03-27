#!/usr/bin/env python3
# A display of death growth rate, offset by intervention.
import argparse
import collections
import csv
import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import pickle

from common import *

parser = argparse.ArgumentParser(description='Plot death growth rates offset by intervention.')
parser.add_argument("-i", "--intervention", default="Lockdown")
parser.add_argument("-m", "--mindeaths", type=int, default=100)
parser.add_argument("-n", "--smoothing", type=int, default=3)
args = parser.parse_args()

# Load the data:
places = pickle.load(open('time_series.pkl', 'rb'))

def csv_as_matrix(path):
    return [r for r in csv.reader(open(path, 'r'))][1:]
country_renames = {r[0]: r[1] for r in csv_as_matrix('data_country_renames.csv')}
place_renames = {
        (r[0],r[1],r[2]): (r[3],r[4],r[5]) for r in csv_as_matrix('data_place_renames.csv')}

intervention_date = {}
for country, region, change, date, explanation in csv_as_matrix('data_interventions.csv'):
    if change == args.intervention:
        if date == '': continue
        p = (country, region, '')
        if p[0] in country_renames: p = (country_renames[p[0]], p[1], p[2])
        if p in place_renames: p = place_renames[p]
        intervention_date[p] = dateutil.parser.parse(date).date()

plots = []

for country in sorted(intervention_date.keys()):
    if country[1] != '': continue
    if country[2] != '': continue
    for province in places.keys():
        if province[2] != '': continue
        if province[1] == '': continue
        if province[0] == country[0]:
            intervention_date[province] = intervention_date[country]



for k in sorted(intervention_date.keys()):
    if k not in places:
        print("Unknown place: ", k, " referenced in interventions.")
        continue
    place_str = ": ".join([s for s in k if s != ''])
    print("place_str:", place_str)
    iv_date = intervention_date[k]
    ts = places[k]
    start_idx = np.argmax(ts.deaths>=args.mindeaths)
    if ts.deaths[start_idx] < args.mindeaths: continue
    dates = ts.dates[start_idx:]
    deaths = ts.deaths[start_idx:]
    for i in range(1, len(deaths)):
        if deaths[i] < deaths[i-1]: deaths[i] = deaths[i-1]
        # Force monotonicity.
    N = args.smoothing
    xs = np.array([(d-iv_date).days for d in dates[:-N]])
    ys = np.power(deaths[N:] / deaths[:-N], 1.0/N) - 1
    plots.append((place_str, xs,ys))

all_vals = collections.defaultdict(list)
for label,xs,ys in plots:
    for x,y in zip(xs, ys):
        all_vals[x].append(y)

average_points = [(x,np.average(all_vals[x])) for x in sorted(all_vals.keys())]
average_xs = np.array([p[0] for p in average_points])
average_ys = np.array([p[1] for p in average_points])
plots.append(('Average', average_xs, average_ys))

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
for label,xs,ys in plots:
    color = 'b'
    if label=='Average': color = 'r'
    ax.plot(xs, ys, color, alpha=0.5, lw=2, label=label)
    ax.set_xlabel('Days (0=intervention)')
    ax.set_ylabel('Growth in deaths')
    ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()

    
