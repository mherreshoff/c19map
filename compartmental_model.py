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
import scipy.stats
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
P_HOSPITAL = 0.137
    # Probability an infectious case gets hospitalized
    # TODO: find citation & tweak.
HOSPITAL_DURATION = 9.75
    # Average length of hospital stay.
    # Note: hospital stays for dying people and recovering people aren't the same length.
    # We use the duration for dying people, because care about the accuracy of
    # the death curve more.
    # 11.2 -> https://www.medrxiv.org/content/10.1101/2020.02.07.20021154v1
    # 8.3 -> https://www.medrxiv.org/content/medrxiv/early/2020/01/28/2020.01.26.20018754.full.pdf
P_DEATH = 0.14
    # Probability of death given hospitaliation.
    # 0.14 -> https://eurosurveillance.org/content/10.2807/1560-7917.ES.2020.25.3.2000044

# Growth rates, used to calculate contact_rate a.k.a. beta(t):
# These get calculated from the input data below.
growth_rate_by_intervention = {}

# Minimum number of deaths needed to tune gowth rates.
empirical_growth_min_deaths = 100

# Largest fraction of the population a growth can be observed at and still be trustworthy.
empirical_growth_max_pop_frac = 0.03

# How many days does an intervention have to be in effect before we consider
# growth data to represent it.
empirical_growth_inv_days = 20

tuned_countries = set(['China', 'Japan', 'Korea, South'])
#tuned_countries = set()

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


# Set up a default version of the model:
model = Model(
        None,
        LATENT_PERIOD,
        INFECTIOUS_PERIOD,
        P_HOSPITAL, HOSPITAL_DURATION,
        P_DEATH)


# Load the JHU time series data:
places = pickle.load(open('time_series.pkl', 'rb'))
population = load_population_data()
def iterate_places():
    for k, ts in sorted(places.items()):
        if k not in population: continue
        N = population[k]
        yield k, N, ts


# Calculate Empirical Growth Rates:
empirical_growths = collections.defaultdict(list)

for k, N, ts in iterate_places():
    country = k[0]
    if country in tuned_countries: continue

    # Get the set of dates after a sufficiently long streak of the same intervention:
    stable_dates = set()
    prev_inv = ts.interventions[0]
    run = 1000
    for inv_d, inv in zip(ts.intervention_dates, ts.interventions):
        if inv != prev_inv: run = 1
        else: run += 1
        prev_inv = inv
        if run >= empirical_growth_inv_days:
            stable_dates.add(inv_d)

    empirical_growths_here = collections.defaultdict(list)
    for date, d, nd in zip(ts.dates, ts.deaths, ts.deaths[1:]):
        if date not in stable_dates: continue
        try:
            inv_idx = ts.intervention_dates.index(date)
            inv = ts.interventions[inv_idx]
        except ValueError:
            continue
        if d < empirical_growth_min_deaths: continue
        if d > N*empirical_growth_max_pop_frac: continue
        growth = nd/d
        # TODO check for being at the start of an intervention.
        empirical_growths_here[inv].append(growth)
    for inv, gs in empirical_growths_here.items():
        empirical_growths[inv].append(scipy.stats.gmean(gs))

for k, gs in sorted(empirical_growths.items()):
    m = np.median(gs)
    growth_rate_by_intervention[k] = m - 1.0
    print('Intervention Status "{k}" has growth rate {m}'.format(k=k,m=m))
growth_rate_by_intervention['Unknown'] = growth_rate_by_intervention['No Intervention']


def interventions_to_gr_by_date(
        iv_dates, iv_strings, growth_rate_power=None):
    # Takes a list of interventions.
    # Returns a function that computes beta from t.
    if growth_rate_power is None:
        growth_rate = growth_rate_by_intervention
    else:
        growth_rate = {}
        for k, gr in growth_rate_by_intervention.items():
            gr = (1 + gr) ** growth_rate_power - 1
            growth_rate[k] = gr
    return {d: growth_rate[s] for d, s in zip(iv_dates, iv_strings)}


def gr_by_date_to_beta_fn(gr_by_date, zero_day):
    t_to_beta = {(d-zero_day).days: seir_growth_rate_to_beta(g)
            for d, g in gr_by_date.items()}
    lowest = min(t_to_beta.keys())
    highest = max(t_to_beta.keys())
    def beta(t):
        t = int(np.floor(t))
        if t < lowest: return t_to_beta[lowest]
        if t > highest: return t_to_beta[highest]
        if t in t_to_beta: return t_to_beta[t]
    return beta



# Outputs:
all_vars_w = csv.writer(open('output_all_vars.csv', 'w'))
all_vars_w.writerow(
        ["Province/State", "Country/Region", "Lat", "Long"] +
        [v for v in Model.variables])


infected_w = csv.writer(open('output_estimated_infected.csv', 'w'))
infected_w.writerow(
        ["Province/State", "Country/Region", "Lat", "Long",
        "Estimated", "Region Population", "Estimated Per Capita"])

growth_rate_w = csv.writer(open('output_limiting_growth_rates.csv', 'w'))
headers = ["Province/State", "Country/Region", "Lat", "Long"]
intervention_dates = list(places.values())[0].intervention_dates
headers += ["%d/%d/%d" % (d.month, d.day, d.year%100)
        for d in intervention_dates]
growth_rate_w.writerow(headers)

# TODO: add flag for whether graphs happen.
graph_output_dir = 'graphs'
if os.path.exists(graph_output_dir):
    shutil.rmtree(graph_output_dir)
os.makedirs(graph_output_dir)

graph_days_forecast = 60  #TODO: flag.
  # How many days into the future do we simulate?

# Totals: for the historytable output.
history_dates = list(places.values())[0].dates
world_confirmed = np.zeros(len(history_dates))
world_deaths = np.zeros(len(history_dates))
world_estimated_cases = np.zeros(len(history_dates))

# Run the model forward for each of the places:
for k, N, ts in iterate_places():
    present_date = ts.dates[-1]
    place_s = ' - '.join([s for s in k if s != ''])
    print("Place =",place_s)

    # We find the first death and start simulating from there
    # with the population set really big.
    nz_deaths, = np.nonzero(ts.deaths)
    if len(nz_deaths) == 0:
        print("No deaths recorded, skipping: ", place_s)
        continue
    start_idx = nz_deaths[0]
    fit_length = len(ts.dates)-start_idx
    gr_by_date = interventions_to_gr_by_date(
            ts.intervention_dates, ts.interventions)
    model.contact_rate = gr_by_date_to_beta_fn(gr_by_date, present_date)
    growth_rate, equilibrium_state = model.equilibrium(t=-(fit_length-1))

    t = np.arange(fit_length) - (fit_length+1)
    target = ts.deaths[start_idx:]
    y0 = equilibrium_state.copy()
    y0[0] = (10**9) - np.sum(y0)

    trajectories = odeint(lambda *a: model.derivative(*a), y0, t)
    S, E, I, H, D, R = trajectories.T
    # TODO: cut off when S < .9*N or some such for accuracy.

    # Then see how much to scale the death data to G
    if k[0] in tuned_countries:
        def loss(x): return np.linalg.norm((D**x[0])*x[1]-target)
        gr_pow, state_scale = scipy.optimize.minimize(
                loss, [1,1], bounds=[(.2, 1), (.01, 100)]).x
        gr_by_date = interventions_to_gr_by_date(
                ts.intervention_dates, ts.interventions, gr_pow)
        model.contact_rate = gr_by_date_to_beta_fn(gr_by_date, present_date)
        growth_rate, equilibrium_state = model.equilibrium(t=-(fit_length-1))
        # Recompute the equilibrium since we've altered the model.
    else:
        def loss(s): return np.linalg.norm(D*s-target)
        state_scale = scipy.optimize.minimize_scalar(loss, bounds=(.01, 100)).x
        gr_pow = None

    present_date = ts.dates[-1]
    days_to_present = len(ts.dates) - 1 - start_idx
    days_simulation = days_to_present + graph_days_forecast + 1
    t = np.arange(days_simulation) - days_to_present
    
    y0 = state_scale * equilibrium_state
    y0[0] = N - np.sum(y0)

    trajectories = odeint(lambda *a: model.derivative(*a), y0, t)
    S, E, I, H, D, R = trajectories.T

    estimated_cases = E+I+H+D+R  # Everyone who's ever been a case.
    # Pad out our growth curve back to time zero with an exponential.
    padding = []
    padding_next = estimated_cases[0]
    for i in range(start_idx):
        padding_next /= growth_rate
        padding.append(padding_next)
    padding.reverse()
    estimated_cases = np.concatenate([padding, estimated_cases])

    # Update world history table:
    world_confirmed += ts.confirmed
    world_deaths += ts.deaths
    world_estimated_cases += estimated_cases[:len(ts.dates)]

    # Variables:
    row_start = [k[0], k[1], ts.latitude, ts.longitude]
    all_vars_w.writerow(row_start + list(np.round(trajectories.T[:,days_to_present])))

    # Estimation:
    latest_estimate = np.round(estimated_cases[len(ts.dates)-1], -3)
    if latest_estimate < 1000: estimated = ''
    infected_w.writerow(row_start + [latest_estimate])

    # Time Sequence for Growth Rates:
    growth_rates = [gr_by_date[d] for d in intervention_dates]
    growth_rate_w.writerow(row_start + growth_rates)

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

# Output world history table:
history_w = csv.writer(open('output_world_history.csv', 'w'))
history_w.writerow([
    "Report Date", "Report Date String", "Confirmed", "Deaths", "Estimated"])

for i, d in enumerate(history_dates):
    short_date = "%d/%d/%d" % (d.month, d.day, d.year%100)
    date_str = d.isoformat()
    confirmed = world_confirmed[i]
    deaths = world_deaths[i]
    estimated_cases = np.round(world_estimated_cases[i])
    history_w.writerow([short_date, date_str, confirmed, deaths, estimated_cases])
