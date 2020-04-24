#!/usr/bin/env python3
import argparse
import collections
import contextlib
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
import sympy
import sys

from common import *


parser = argparse.ArgumentParser(description=\
        """Read the intervention data and run our model on it, outputting predictions.""")
# Model parameters:
parser.add_argument("--latent_period", "--lt", default=3.5, type=float)
    # 1/sigma - The average length of time between contracting the disease
    #     and becoming infectious.
    # Citation: 3 studies in the Midas Database with fairly close agreement.
parser.add_argument("--infectious_period", "--it", default=4.2, type=float)
    # 1/gamma - The average length of time a person stays infectious
    # Currently an average of MIDAS Database results for "Time From Symptom Onset To Hospitalization".
    # Heuristic approximation because infectious period may start before symptoms.
parser.add_argument("--p_hospital", "--hp", default=0.0714, type=float)
    # Probability an infectious case gets hospitalized
    # We divided the IFR estimate by the probability of death given hospitalization, below, to get the probability of
    # hospitalization.
parser.add_argument("--hospital_duration", "--ht", default=9.75, type=float)
    # Average length of hospital stay.
    # Note: hospital stays for dying people and recovering people aren't the same length.
    # We use the duration for dying people, because care about the accuracy of
    # the death curve more.
    # 11.2 -> https://www.medrxiv.org/content/10.1101/2020.02.07.20021154v1
    # 8.3 -> https://www.medrxiv.org/content/medrxiv/early/2020/01/28/2020.01.26.20018754.full.pdf
parser.add_argument("--p_death_given_hospital", "--dp", default=0.14, type=float)
    # Probability of death given hospitaliation.
    # 0.14 -> https://eurosurveillance.org/content/10.2807/1560-7917.ES.2020.25.3.2000044

# Empirical growth detection parameters:
parser.add_argument("--empirical_growth_min_deaths", default=100, type=int)
    # Minimum number of deaths needed to tune gowth rates.

parser.add_argument("--empirical_growth_max_pop_frac", default=0.03, type=float)
    # Largest fraction of the population a growth can be observed at and still be trustworthy.

parser.add_argument("--empirical_growth_inv_days", default=20, type=int)
    # How many days does an intervention have to be in effect before we consider
    # growth data to represent it.

#Others:
parser.add_argument("--optimize_lockdown", default=True, type=bool)
    # Attempts to run an optimization to find out how beta changes over a typical lockdown.

parser.add_argument("--debug_lockdown_fit", default=False, type=bool)
    # Shows a graph of how our lockdown betas fit the data.

parser.add_argument('--tuned_countries',
        default=['China', 'Japan', 'Korea, South'], nargs='*')
    # The countries we treat specially

parser.add_argument('-c', '--countries', default=[], nargs='*')
    # The countries we run the simulation for.  (Unspecified means all countries.)

args = parser.parse_args()

class Model:
    variables = "SEIHDR"
    var_to_id = {v: i for i, v in enumerate(variables)}
    variable_names = ['Susceptible', 'Exposed', 'Infected',
            'Hospitalized', 'Dead', 'Recovered']
    # The SEIHDR Model.
    # An extension of the SEIR model.
    # TODO link to the google doc here.
    def __init__(self,
            contact_rate, latent_t, infectious_t,
            hospital_p, hospital_t, death_p):
        self.contact_rate = contact_rate
        self.latent_t = latent_t
        self.infectious_t = infectious_t
        self.hospital_p = hospital_p
        self.hospital_t = hospital_t
        self.death_p = death_p

    def param_str(self):
        return (f"LT={self.latent_t} IT={self.infectious_t} " +
        f"HP={self.hospital_p} HT={self.hospital_t} DP={self.death_p}")

    @contextlib.contextmanager
    def beta(self, val):
        old_contact_rate = self.contact_rate
        if callable(val): self.contact_rate = val
        else: self.contact_rate = constant_fn(val)
        try: yield None
        finally: self.contact_rate = old_contact_rate

    def companion_matrix(self, t=0, correction=1, dtype=float):
        """ Companion matrix for the linear approximation of the model.

        Put in S/N for the correction term to get a matrix which computes
        the gradient of the non-linearized model.
        """
        flows = [  # (from variable, to variable, amount variable, rate)
            ('S','E', 'I', self.contact_rate(t)*correction),
            ('E','I', 'E', 1/self.latent_t),
            ('I','H', 'I', (1/self.infectious_t)*self.hospital_p),
            ('I','R', 'I', (1/self.infectious_t)*(1-self.hospital_p)),
            ('H','D', 'H', (1/self.hospital_t)*self.death_p),
            ('H','R', 'H', (1/self.hospital_t)*(1-self.death_p))]
        nv = len(Model.variables)
        m = np.zeros((nv, nv), dtype=dtype)
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

    @functools.lru_cache(maxsize=10000)
    def beta_to_growth_rate(self, beta):
        with self.beta(beta):
            m = self.companion_matrix(dtype=float)
            m = m[1:4,1:4] # Get rid of S, D, and R variables.
            w, v = np.linalg.eig(m)
            max_eig_id = int(np.argmax(w))
            growth_rate = np.exp(w[max_eig_id])
            return growth_rate

    @functools.lru_cache(maxsize=10000)
    def growth_rate_to_beta(self, growth_rate):
        target_eigenval = np.log(growth_rate)
        beta = sympy.Symbol('beta')
        with self.beta(beta):
            m = self.companion_matrix(dtype=object)
            m = m[1:4,1:4] # Get rid of S,D, and R variables.
            m = sympy.Matrix(m)
            eigenvals = list(m.eigenvals().keys())
                # Symbolic expressions in terms of beta.
            solutions = []
            for eigenval in eigenvals:
                solutions += sympy.solvers.solve(
                        eigenval-target_eigenval, beta)
            assert len(solutions) == 1
            return solutions[0]

    def derivative(self, y, t):
        S,E,I,H,D,R = y
        N = np.sum(y)
        m = self.companion_matrix(t=t, correction=S/N)
        return np.matmul(m, y)


# Set up a default version of the model:
model = Model(
        None,
        args.latent_period,
        args.infectious_period,
        args.p_hospital, args.hospital_duration,
        args.p_death_given_hospital)


# Load the JHU time series data:
places = pickle.load(open('time_series.pkl', 'rb'))


# Growth rates, used to calculate betas.
fixed_growth_by_inv = {}

# Calculate Empirical Growth Rates:
empirical_growths = collections.defaultdict(list)


for p in places.values():
    if p.population is None: continue
    if p.country in args.tuned_countries: continue

    # Get the set of dates after a sufficiently long streak of the same intervention:
    stable_dates = set()
    prev_inv = p.interventions[0]
    run = 1000
    for inv_d, inv in p.interventions.items():
        if inv != prev_inv: run = 1
        else: run += 1
        prev_inv = inv
        if run >= args.empirical_growth_inv_days:
            stable_dates.add(inv_d)

    empirical_growths_here = collections.defaultdict(list)
    for date, d, nd in zip(
            p.deaths.dates(), p.deaths.array(), p.deaths.array()[1:]):
        if date not in stable_dates: continue
        if d < args.empirical_growth_min_deaths: continue
        if d > p.population*args.empirical_growth_max_pop_frac: continue
        inv = p.interventions[date]
        growth = nd/d
        empirical_growths_here[inv].append(growth)
    for inv, gs in empirical_growths_here.items():
        empirical_growths[inv].append(scipy.stats.gmean(gs))

for k, gs in sorted(empirical_growths.items()):
    m = np.median(gs)
    fixed_growth_by_inv[k] = m
    print(f"Intervention Status \"{k}\" has growth rate {m}")
fixed_growth_by_inv['Unknown'] = fixed_growth_by_inv['No Intervention']

beta_by_intervention = {}
for k, v in fixed_growth_by_inv.items():
    beta = model.growth_rate_to_beta(v)
    print(f"k={k}: v={v} -> beta={beta}")
    beta_by_intervention[k] = constant_fn(beta)

deaths_rel_to_lockdown = collections.defaultdict(list)
for p in places.values():
    if p.interventions is None: continue
    first_lockdown_date = p.interventions.date_of_first('Lockdown')
    if first_lockdown_date is None: continue
    if first_lockdown_date not in p.deaths.dates(): continue
    lockdown_idx = p.deaths.date_to_position(first_lockdown_date)
    deaths_at_lockdown = p.deaths[first_lockdown_date]
    if deaths_at_lockdown < 5: continue
    for i, d in enumerate(p.deaths):
        deaths_rel_to_lockdown[i-lockdown_idx].append(d/deaths_at_lockdown)

lockdown_death_trend = []
for k, v in sorted(deaths_rel_to_lockdown.items()):
    if k < 0: continue
    if len(v) < 5: break
    lockdown_death_trend.append(np.mean(v))

default_beta = model.growth_rate_to_beta(fixed_growth_by_inv['No Intervention'])
with model.beta(default_beta):
    no_inv_gr, y0 = model.equilibrium()
y0[0] = 1000000000
ts = np.arange(len(lockdown_death_trend))


def lockdown_curve_beta(params):
    def beta(t):
        return np.interp(t, [0, 14], params)
    return beta


def lockdown_curve_fit_traj(params):
    with model.beta(lockdown_curve_beta(params)):
        return odeint(lambda *a: model.derivative(*a), y0, ts)


def lockdown_curve_fit(params):
    S, E, I, H, D, R = lockdown_curve_fit_traj(params).T
    diff = np.linalg.norm(D - np.array(lockdown_death_trend, dtype=float))
    return diff

if args.optimize_lockdown:
    lockdown_curve_params = scipy.optimize.minimize(
            lockdown_curve_fit,
            np.array([model.growth_rate_to_beta(1.2)]*2, dtype=float)).x
    for b,i in enumerate(lockdown_curve_params):
        g = model.beta_to_growth_rate(b)
        print(f"\tLockdown: β_{i} = {b} ... growth={g}")
    beta_by_intervention['Lockdown'] = lockdown_curve_beta(lockdown_curve_params)

for k, b in beta_by_intervention.items():
    print(f"{k} -> β(0)={b(0)} ... β(10)={b(10)}")


if args.debug_lockdown_fit:
    trajectories = lockdown_curve_fit_traj(lockdown_curve_params)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    ax.set_title("Model Fit to Lockdown Death Trend")
    ax.set_xlabel('Days (Lockdown at 0)')
    ax.set_ylabel('People (log)')
    for var, curve in list(zip(Model.variables, trajectories.T))[1:]:
        ax.semilogy(ts, curve, label=var)
    ax.semilogy(ts, lockdown_death_trend, 's', label='D trend.')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()
    sys.exit(0)


def interventions_to_beta_fn(
        iv, zero_day, growth_rate_power=None):
    beta_starts = []
    prev_s = ''
    for d, s in iv.items():
        if s != prev_s:
            beta_starts.append(
                    ((d-zero_day).days, beta_by_intervention[s]))
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
        return model.growth_rate_to_beta(
                model.beta_to_growth_rate(b)**growth_rate_power)
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
intervention_dates = list(list(places.values())[0].interventions.dates())
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
history_dates = list(list(places.values())[0].deaths.dates())
world_confirmed = np.zeros(len(history_dates))
world_deaths = np.zeros(len(history_dates))
world_estimated_cases = np.zeros(len(history_dates))

# Run the model forward for each of the places:

for k, p in sorted(places.items()):
    N = p.population
    if N is None: continue
    if args.countries and k[0] not in args.countries: continue

    present_date = p.deaths.last_date()
    print("Place =", p.region_id())

    # We find the first death and start simulating from there
    # with the population set really big.
    nz_deaths, = np.nonzero(p.deaths)
    if len(nz_deaths) == 0:
        print("No deaths recorded, skipping: ", p.region_id())
        continue
    start_idx = nz_deaths[0]
    fit_length = len(p.deaths)-start_idx
    beta = interventions_to_beta_fn(p.interventions, present_date)
    with model.beta(beta):
        growth_rate, equilibrium_state = model.equilibrium(t=-(fit_length-1))
        t = np.arange(fit_length) - (fit_length+1)
        target = p.deaths[start_idx:]
        y0 = equilibrium_state.copy()
        y0[0] = (10**9) - np.sum(y0)

        trajectories = odeint(lambda *a: model.derivative(*a), y0, t)
        S, E, I, H, D, R = trajectories.T
    # TODO: cut off when S < .9*N or some such for accuracy.

    # Then see how much to scale the death data to G
    if k[0] in args.tuned_countries:
        def loss(x): return np.linalg.norm((D**x[0])*x[1]-target)
        gr_pow, state_scale = scipy.optimize.minimize(
                loss, [1,1], bounds=[(.2, 1), (.01, 100)]).x
        beta = interventions_to_beta_fn(
                p.interventions, present_date, gr_pow)
        with model.beta(beta):
            growth_rate, equilibrium_state = model.equilibrium(t=-(fit_length-1))
        # Recompute the equilibrium since we've altered the model.
    else:
        def loss(s): return np.linalg.norm(D*s-target)
        state_scale = scipy.optimize.minimize_scalar(loss, bounds=(.01, 100)).x
        gr_pow = None

    present_date = p.deaths.last_date()
    days_to_present = len(p.deaths) - 1 - start_idx
    days_simulation = days_to_present + graph_days_forecast + 1
    t = np.arange(days_simulation) - days_to_present
    
    y0 = state_scale * equilibrium_state
    y0[0] = N - np.sum(y0)

    with model.beta(beta):
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
    world_confirmed += p.confirmed
    world_deaths += p.deaths
    world_estimated_cases += estimated_cases[:len(p.deaths)]

    # Output Estimate:
    present_est = np.round(estimated_cases[len(p.deaths)-1])
    print(f"{p.region_id()}\t{model.param_str()}\t{present_est}")

    # Output all variables:
    row_start = [k[0], k[1], p.latitude, p.longitude]
    all_vars_w.writerow(row_start + list(np.round(trajectories.T[:,days_to_present])))

    # Output Estimation:
    latest_estimate = np.round(estimated_cases[len(p.deaths)-1], -3)
    if latest_estimate < 1000: estimated = ''
    infected_w.writerow(row_start + [latest_estimate])

    # Output Time Sequence for Growth Rates:
    growth_rates = [
            model.beta_to_growth_rate(beta((d-present_date).days))
            for d in p.interventions.dates()]
    growth_rate_w.writerow(row_start + growth_rates)

    # Output comprehensive CSVs:
    country, province, district = k
    prev_jhu_fields = None
    prev_estimated_fields = None
    for idx, d in enumerate(p.deaths.dates()):
        intervention = p.interventions.extrapolate(d)

        initial_fields = [p.region_id(), d.isoformat(),
                p.display_name(), country, province,
                p.latitude, p.longitude,
                intervention, p.population]
        def per10k(x):
            if x == '': return ''
            return np.round(10000*x/p.population)
        def round_delta(x, y):
            if x == '' or y == '': return ''
            return np.round(x-y)

        jhu_fields = [p.confirmed[idx], p.deaths[idx]]
        jhu_per10k_fields = [per10k(s) for s in jhu_fields]
        if prev_jhu_fields is None: jhu_delta_fields = ['']*2
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
        estimated_per10k_fields = [per10k(s) for s in estimated_fields]
        if prev_estimated_fields is None:
            estimated_delta_fields = ['']*len(estimated_fields)
        else: estimated_delta_fields = [round_delta(x, p)
                for x,p in zip(estimated_fields, prev_estimated_fields)]
        prev_estimated_fields = estimated_fields
        estimated_fields = [friendly_round(s) for s in estimated_fields]
        all_stat_fields = (
                jhu_fields + estimated_fields +
                jhu_delta_fields + estimated_delta_fields +
                jhu_per10k_fields + estimated_per10k_fields)
        all_stat_fields = [(0 if x == '' else x) for x in all_stat_fields]
        misc_fields = [present_date.isoformat(), '', '']
        all_fields = initial_fields + all_stat_fields + misc_fields
        output_comprehensive_series_w.writerow(all_fields)
        if d == present_date:
            output_comprehensive_snapshot_w.writerow(all_fields)

    # Graphs:
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    intervention_starts = []
    old_s = 'No Intervention'
    for d,s in p.interventions.items():
        if s!=old_s: intervention_starts.append((d, s))
        old_s = s
    intervention_s = ', '.join(
            s+" on "+d.isoformat() for d,s in intervention_starts)
    ax.set_title(p.region_id() + "\n" + intervention_s)
    ax.set_xlabel('Days (0 is '+present_date.isoformat()+')')
    ax.set_ylabel('People (log)')
    for var, curve in zip(Model.variables, trajectories.T):
        ax.semilogy(t, curve, label=var)
    ax.semilogy(t[0:days_to_present+1], p.deaths[start_idx:],
            's', label='D emp.')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.savefig(os.path.join('graphs', p.region_id() + '.png'), dpi=300)
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
