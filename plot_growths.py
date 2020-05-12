#!/usr/bin/env python3
# A display of death growth rate, offset by intervention.
import argparse
import collections
import csv
import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='Plot death growth rates offset by intervention.')
parser.add_argument("-i", "--intervention", default="Lockdown")
parser.add_argument("-m", "--mindeaths", type=int, default=100)
parser.add_argument("-n", "--smoothing", type=int, default=3)
args = parser.parse_args()

# Load the data:
places = pickle.load(open('places.pkl', 'rb'))

intervention_date = {}
for k, place in places.items():
    print(k)
    inv_date = place.interventions.date_of_first(args.intervention)
    if inv_date is None: continue
    intervention_date[k] = inv_date

for country in sorted(intervention_date.keys()):
    if country[1] != '' or country[2] != '': continue
    for province in places.keys():
        if province in intervention_date: continue
        if province[0] == country[0] and province[1] != '' and province[2] == '':
            intervention_date[province] = intervention_date[country]

plots = []

for k, iv_date in sorted(intervention_date.items()):
    place_str = ": ".join([s for s in k if s != ''])
    print("place_str:", place_str)
    place = places[k]
    start_idx = np.argmax(place.deaths.array()>=args.mindeaths)
    if place.deaths[start_idx] < args.mindeaths: continue
    dates = list(place.deaths.dates())[start_idx:]
    deaths = place.deaths[start_idx:]
    for i in range(1, len(deaths)):
        if deaths[i] < deaths[i-1]: deaths[i] = deaths[i-1]
        # Force monotonicity.
    N = args.smoothing
    xs = np.array([(d-iv_date).days for d in dates[:-N]])
    ys = np.power(deaths[N:] / deaths[:-N], 1.0/N) - 1
    plots.append((place_str, xs, ys))

all_vals = collections.defaultdict(list)
for label,xs,ys in plots:
    for x,y in zip(xs, ys):
        all_vals[x].append(y)

all_vals = {x: y for x, y in all_vals.items() if len(y) >= 3}

average_points = [(x,np.average(all_vals[x])) for x in sorted(all_vals.keys())]
average_xs = np.array([p[0] for p in average_points])
average_ys = np.array([p[1] for p in average_points])

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.set_title("Death Growths After Intervention")
ax.set_xlabel('Days (0=intervention)')
ax.set_ylabel('Growth in deaths')
ax.xaxis.set_tick_params(length=0)
xs = sorted(all_vals.keys())
ax.boxplot([all_vals[x] for x in xs], positions=xs)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()

