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


time_program_began = datetime.datetime.now().isoformat()


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


# Minimum number of deaths needed to tune gowth rates.
empirical_growth_min_deaths = 100

# Largest fraction of the population a growth can be observed at and still be trustworthy.
empirical_growth_max_pop_frac = 0.03

# How many days does an intervention have to be in effect before we consider
# growth data to represent it.
empirical_growth_inv_days = 20

# Set to True to see a graph of how our lockdown betas fit the data.
DEBUG_LOCKDOWN_FIT = False

# The countries we treat specially
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
    return np.exp(np.max(w))


@functools.lru_cache(maxsize=10000)
def seir_growth_rate_to_beta(igr):
    sigma = 1/LATENT_PERIOD
    gamma = 1/INFECTIOUS_PERIOD
    beta = sp.Symbol('beta')
    target_eigenval = np.log(igr)
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
            ('H','D', 'H', (1/self.hospital_t)*self.death_p),
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
        m = self.companion_matrix(t)
        m = m[1:,1:]
            # Get rid of the 'S' variable.  Equilibrium only makes sense if we're
            # assuming an infinite population to expand into.
        w, v = np.linalg.eig(m)
        max_eig_id = int(np.argmax(w))
        growth_rate = np.exp(w[max_eig_id])
        state = v[:,max_eig_id]
        state = np.concatenate([[0], state])  # Add back S row.
        state /= state[Model.variables.index('D')] # Normalize by deaths.
        return growth_rate, state

    def derivative(self, y, t):
        S,E,I,H,D,R = y
        N = np.sum(y) - D
        flows = [  # (from variable, to variable, rate)
            ('S','E', self.contact_rate(t) * I * S / N),
            ('E','I', E/self.latent_t),
            ('I','H', (I/self.infectious_t)*self.hospital_p),
            ('I','R', (I/self.infectious_t)*(1-self.hospital_p)),
            ('H','D', (H/self.hospital_t)*self.death_p),
            ('H','R', (H/self.hospital_t)*(1-self.death_p))]
        inbound = np.array([
            sum(x for s,t,x in flows if t==v) for v in Model.variables])
        outbound = np.array([
            sum(x for s,t,x in flows if s==v) for v in Model.variables])
        return list(inbound-outbound)

    def deaths_to_betas(self, D):
        """Infers a time series of betas using a time series of deaths.

        Starts by using differential equation for deaths to infer hospitalizations.
        Continues backwards from there.

        NOTE: Doesn't seem to work very well.
        """
        def d(xs):
            d = xs[1:]-xs[:-1]
            d = np.pad(d, (1,1), 'edge')
            return (d[:-1]+d[1:])/2.0
        H = d(D) * self.hospital_t / self.death_p
        I = d(H) * self.infectious_t / self.hospital_p
        E = (d(I) + (I / self.infectious_t)) * self.latent_t
        beta = (d(E) + E / self.latent_t) / np.maximum(I, 0.1)
            # The np.max prevents divisions by zero.
            # Note: for now we assume S~N here.
        return beta



# Set up a default version of the model:
model = Model(
        None,
        LATENT_PERIOD,
        INFECTIOUS_PERIOD,
        P_HOSPITAL, HOSPITAL_DURATION,
        P_DEATH)


# Load the JHU time series data:
places = pickle.load(open('time_series.pkl', 'rb'))


# Growth rates, used to calculate contact_rate.
fixed_growth_by_inv = {}

# Calculate Empirical Growth Rates:
empirical_growths = collections.defaultdict(list)


for k, ts in sorted(places.items()):
    N = ts.population
    if N is None: continue
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
    fixed_growth_by_inv[k] = m
    print('Intervention Status "{k}" has growth rate {m}'.format(k=k,m=m))
fixed_growth_by_inv['Unknown'] = fixed_growth_by_inv['No Intervention']

beta_by_intervention = {}
for k, v in fixed_growth_by_inv.items():
    beta = seir_growth_rate_to_beta(v)
    beta_by_intervention[k] = lambda t: beta

deaths_rel_to_lockdown = collections.defaultdict(list)
for k, ts in sorted(places.items()):
    if 'Lockdown' not in ts.interventions: continue
    first_lockdown_ts_idx = ts.interventions.index('Lockdown')
    first_lockdown_date = ts.intervention_dates[first_lockdown_ts_idx]
    if first_lockdown_date not in ts.dates: continue
    lockdown_idx = ts.dates.index(first_lockdown_date)
    deaths_at_lockdown = ts.deaths[lockdown_idx]
    if deaths_at_lockdown < 5: continue
    for i, d in enumerate(ts.deaths):
        deaths_rel_to_lockdown[i-lockdown_idx].append(d/deaths_at_lockdown)

lockdown_death_trend = []
for k, v in sorted(deaths_rel_to_lockdown.items()):
    if k < 0: continue
    if len(v) < 5: break
    lockdown_death_trend.append(np.mean(v))

default_beta = seir_growth_rate_to_beta(fixed_growth_by_inv['No Intervention'])
model.contact_rate = lambda t: default_beta
no_inv_gr, y0 = model.equilibrium()
y0[0] = 1000000000
ts = np.arange(len(lockdown_death_trend))


def lockdown_curve_beta(params):
    def beta(t):
        return np.interp(t, [0, 14], params)
    return beta


def lockdown_curve_fit_traj(params):
    model.contact_rate = lockdown_curve_beta(params)
    trajectories = odeint(lambda *a: model.derivative(*a), y0, ts)
    return trajectories


def lockdown_curve_fit(params):
    S, E, I, H, D, R = lockdown_curve_fit_traj(params).T
    diff = np.linalg.norm(D - np.array(lockdown_death_trend, dtype=float))
    return diff


lockdown_curve_params = scipy.optimize.minimize(
        lockdown_curve_fit,
        np.array([seir_growth_rate_to_beta(1.2)]*2, dtype=float)).x

print()
for i, b in enumerate(lockdown_curve_params):
    print("    beta{i} = {b} --> growth rate = {g}".format(
        i=i, b=b, g=seir_beta_to_growth_rate(b)))

# Use the curve we got from the optimization for the lockdown category.
beta_by_intervention['Lockdown'] = lockdown_curve_beta(lockdown_curve_params)


if DEBUG_LOCKDOWN_FIT:
    trajectories = lockdown_curve_fit_traj(lockdown_curve_params)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    ax.set_title("Places post-intervention")
    ax.set_xlabel('Days (0 is intervention)')
    ax.set_ylabel('People (log)')
    for var, curve in list(zip(Model.variables, trajectories.T))[1:]:
        ax.semilogy(ts, curve, label=var)
    ax.semilogy(ts, lockdown_death_trend, 's', label='D trend.')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()


def interventions_to_beta_fn(
        iv_dates, iv_strings, zero_day, growth_rate_power=None):
    beta_starts = []
    prev_s = ''
    for d, s in zip(iv_dates, iv_strings):
        if s != prev_s:
            beta_starts.append(
                    ((d-zero_day).days,
                        beta_by_intervention[s]))
            prev_s = s
    beta_starts.sort(reverse=True)
    def beta_fn(t):
        for s, f in beta_starts:
            if t >= s:
                b = f(t - s)
                break
        else:
            s, f = beta_starts[-1]
            b = f(0)
        if growth_rate_power is None: return b
        return seir_growth_rate_to_beta(
                seir_beta_to_growth_rate(b)**growth_rate_power)
    return beta_fn


# Initialize CSV Outputs:
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


output_comprehensive_series_w = csv.writer(
        open('output_comprehensive_series.csv', 'w'))
output_comprehensive_snapshot_w = csv.writer(
        open('output_comprehensive_snapshot.csv', 'w'))
headers = ["Region ID", "Date",
        "Display Name", "Country/Region", "Province/State",
        "Latitude", "Longitude",
        "Intervention Status",
        "Population"]
stat_columns = ["Reported Confirmed","Reported Deaths",
        "Susceptible","Exposed","Infectious","Hospitalized","Dead","Recovered",
        "Active","Cumulative Infected"]
headers += stat_columns
headers += [s + " (delta)" for s in stat_columns]
headers += [s + " (per 10k)" for s in stat_columns]
headers += ["Last Updated", "Message", "Notes"]
output_comprehensive_series_w.writerow(headers)
headers[1] = "Snapshot Date"
output_comprehensive_snapshot_w.writerow(headers)


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

for k, ts in sorted(places.items()):
    N = ts.population
    if N is None: continue

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
    model.contact_rate = interventions_to_beta_fn(
            ts.intervention_dates, ts.interventions, present_date)
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
        model.contact_rate = interventions_to_beta_fn(
            ts.intervention_dates, ts.interventions, present_date, gr_pow)
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

    # Output all variables:
    row_start = [k[0], k[1], ts.latitude, ts.longitude]
    all_vars_w.writerow(row_start + list(np.round(trajectories.T[:,days_to_present])))

    # Output Estimation:
    latest_estimate = np.round(estimated_cases[len(ts.dates)-1], -3)
    if latest_estimate < 1000: estimated = ''
    infected_w.writerow(row_start + [latest_estimate])

    # Output Time Sequence for Growth Rates:
    growth_rates = [
            seir_beta_to_growth_rate(model.contact_rate((d-present_date).days))
            for d in intervention_dates]
    growth_rate_w.writerow(row_start + growth_rates)

    # Comprehensive output CSV:
    region_id = ' - '.join(s for s in k if s != '')
    def place_display_name(place):
        parts = list(collections.OrderedDict.fromkeys(place))
        return ' - '.join(p for p in parts if p != '')
    display_name = place_display_name(k)

    country, province, district = k
    prev_jhu_fields = None
    prev_estimated_fields = None
    for idx, d in enumerate(ts.dates):
        if d in ts.intervention_dates:
            intervention = ts.interventions[ts.intervention_dates.index(d)]
        elif d < ts.intervention_dates[0]:
            intervention = ts.interventions[0]
        elif d > ts.intervention_dates[-1]:
            intervention = ts.interventions[-1]
        else: assert False, "Couldn't find intervention."

        initial_fields = [region_id, d.isoformat(),
                display_name, country, province,
                ts.latitude, ts.longitude,
                intervention, ts.population]
        def per10k(x):
            if x == '': return ''
            return np.round(10000*x/ts.population)
        def round_delta(x, y):
            if x == '' or y == '': return ''
            return np.round(x-y)

        jhu_fields = [ts.confirmed[idx], ts.deaths[idx]]
        jhu_per10k_fields = [per10k(s) for s in jhu_fields]
        if prev_jhu_fields is None: jhu_delta_fields = ['', '']
        else: jhu_delta_fields = [round_delta(x,p)
                for x,p in zip(jhu_fields, prev_jhu_fields)]
        prev_jhu_fields = jhu_fields

        if idx >= start_idx:
            sim_idx = idx-start_idx
            estimated_fields = list(trajectories[sim_idx])
            active_infections = I[sim_idx]+E[sim_idx]+H[sim_idx]
            cumulative_infections = active_infections + R[sim_idx] + D[sim_idx]
            estimated_fields += [active_infections, cumulative_infections]
        else:
            estimated_fields = ['']*8
        def round_thousands(x):
            if x == '': return ''
            return np.round(x, -3)
        estimated_per10k_fields = [per10k(s) for s in estimated_fields]
        if prev_estimated_fields is None:
            estimated_delta_fields = ['']*len(estimated_fields)
        else: estimated_delta_fields = [round_delta(x, p)
                for x,p in zip(estimated_fields, prev_estimated_fields)]
        prev_estimated_fields = estimated_fields
        estimated_fields = [round_thousands(s) for s in estimated_fields]
        misc_fields = [present_date.isoformat(), '', '']
        all_fields = (
                initial_fields +
                jhu_fields + estimated_fields +
                jhu_delta_fields + estimated_delta_fields +
                jhu_per10k_fields + estimated_per10k_fields +
                misc_fields)
        output_comprehensive_series_w.writerow(all_fields)
        if d == present_date:
            output_comprehensive_snapshot_w.writerow(all_fields)

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
