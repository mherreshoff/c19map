#!/usr/bin/env python3
import argparse
from bunch import Bunch
import collections
import contextlib
import csv
import datetime
import dateutil.parser
import functools
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy
import scipy.stats
import shutil
import sympy
import sympy.utilities.lambdify
import sys

from common import *
import integrators


# --------------------------------------------------------------------------------
# Flags
parser = argparse.ArgumentParser(description=\
        """Read the intervention data and run our model on it, outputting predictions.""")

# Region selection:
parser.add_argument('-c', '--countries', default=[], nargs='*')
parser.add_argument('-p', '--places', default=[], nargs='*')
    # The countries/places we run the simulation for.
    # If neither is specified we run on everything.

# Model parameter flags:
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

# Empirical growth detection flags:
parser.add_argument("--empirical_growth_min_deaths", default=20, type=int)
    # Minimum number of deaths needed to tune gowth rates.

parser.add_argument("--empirical_growth_max_pop_frac", default=0.03, type=float)
    # Largest fraction of the population a growth can be observed at and still be trustworthy.

parser.add_argument("--empirical_growth_min_inv_days", default=20, type=int)
    # How many days does an intervention have to be in effect before we consider
    # growth data to represent it.

# Lockdown fitting flags:
parser.add_argument("--optimize_lockdown", default=True, type=bool)
    # Attempts to run an optimization to find out how beta changes over a typical lockdown.

parser.add_argument("--lockdown_warmup", default=28, type=int)
    # How many days it takes for a lockdown to reach 100% effect.

parser.add_argument("--display_curve_fit", action='store_true')
    # Shows a graph of how our lockdown betas fit the data.


# Graph flags:
parser.add_argument("--nograph", action='store_true')
    # Turn off graphs.

parser.add_argument("--graph_days_forecast", default=60, type=int)
    # How far into the future to draw the graphs.

parser.add_argument("--graph_back", action='store_true')
    # Attempts to run an optimization to find out how beta changes over a typical lockdown.

parser.add_argument("--graph_bottom", action='store_true')
    # Shows y-values < 1 in the graph outputs.

args = parser.parse_args()

# --------------------------------------------------------------------------------
# Model

class Model:
    variables = "SEIHDR"
    var_to_id = {v: i for i, v in enumerate(variables)}
    variable_names = ['Susceptible', 'Exposed', 'Infectious',
            'Hospitalized', 'Dead', 'Recovered']
    # The SEIHDR Model.
    # An extension of the SEIR model.
    # See: https://bit.ly/c19map-v2-docs
    def __init__(self,
            contact_rate, latent_t, infectious_t,
            hospital_p, hospital_t, death_p):
        self.contact_rate = contact_rate
        self.latent_t = latent_t
        self.infectious_t = infectious_t
        self.hospital_p = hospital_p
        self.hospital_t = hospital_t
        self.death_p = death_p

        self.eigen_value_to_beta = None

    def param_str(self):
        return (f"LT={self.latent_t} IT={self.infectious_t} " +
        f"HP={self.hospital_p} HT={self.hospital_t} DP={self.death_p}")

    @contextlib.contextmanager
    def beta(self, val):
        old_contact_rate = self.contact_rate
        if callable(val):
            self.contact_rate = val
        else:
            self.contact_rate = constant_fn(val)
            try:
                setattr(self.contact_rate, 'betas', np.array([val], dtype=float))
                setattr(self.contact_rate, 'ts', np.array([0], dtype=float))
            except Exception:
                pass
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

    @functools.lru_cache(maxsize=100000)
    def beta_to_growth_rate(self, beta):
        with self.beta(beta):
            m = self.companion_matrix(dtype=float)
            m = m[1:4,1:4] # Get rid of S, D, and R variables.
            w, v = np.linalg.eig(m)
            max_eig_id = int(np.argmax(w))
            growth_rate = np.exp(w[max_eig_id])
            return growth_rate

    @functools.lru_cache(maxsize=100000)
    def growth_rate_to_beta(self, growth_rate):
        if not self.eigen_value_to_beta:
            x = sympy.Symbol('x')  # eigenvalue (log growth-rate)
            beta = sympy.Symbol('beta')
            with self.beta(beta):
                m = self.companion_matrix(dtype=object)
            m = m[1:4,1:4] # Get rid of S,D, and R variables.
            m = sympy.Matrix(m)
            eigenvals = list(m.eigenvals().keys())
            solutions = []
            for e in eigenvals:
                solutions += sympy.solvers.solve(e-x, beta)
            assert len(solutions) == 1
            self.eigen_value_to_beta = sympy.utilities.lambdify(x, solutions[0])
        return self.eigen_value_to_beta(np.log(growth_rate))

    def derivative(self, t, y):
        S,E,I,H,D,R = y
        N = np.sum(y) - D
        m = self.companion_matrix(t=t, correction=S/N)
        return np.matmul(m, y)

    def integrate(self, y0, ts, use_cython=True):
        if use_cython:
            return self.integrate_cython(y0, ts)
        else: f = self.integrate_scipy
        return f(y0, ts)

    def integrate_scipy(self, y0, ts):
        soln = scipy.integrate.solve_ivp(
                fun=lambda *a: self.derivative(*a),
                t_span=(min(ts), max(ts)),
                y0=y0,
                t_eval=ts)
        return soln.y.T

    def integrate_cython(self, y0, ts):
        params = np.array([
            1/self.latent_t,
            1/self.infectious_t,
            1/self.hospital_t,
            self.hospital_p,
            self.death_p], dtype=float)
        ts = np.array(ts, dtype=float)
        beta_ts = self.contact_rate.ts
        beta_vals = self.contact_rate.betas
        trajectory = integrators.augmented_seir_integrate(
                y0, params, beta_ts, beta_vals, ts, 0.10)
        return trajectory

# --------------------------------------------------------------------------------
# Calculation of growths, trends, betas, etc. with which to tune the model.

def calculate_empirical_growths(places, min_deaths, min_inv_days, max_pop_frac):
    empirical_growths = collections.defaultdict(list)
    for k, p in sorted(places.items()):
        if p.population is None: continue

        # Get the set of dates after a sufficiently long streak of the same intervention:
        stable_dates = set()
        prev_inv = p.interventions[0]
        run = 1000
        for inv_d, inv in p.interventions.items():
            if inv != prev_inv: run = 1
            else: run += 1
            prev_inv = inv
            if run >= min_inv_days:
                stable_dates.add(inv_d)

        empirical_growths_here = collections.defaultdict(list)
        for date, d, nd in zip(
                p.deaths.dates(), p.deaths.array(), p.deaths.array()[1:]):
            if date not in stable_dates: continue
            if d < min_deaths: continue
            if d > p.population*max_pop_frac: continue
            inv = p.interventions[date]
            growth = nd/d
            empirical_growths_here[inv].append(growth)
        for inv, gs in empirical_growths_here.items():
            m = scipy.stats.gmean(gs)
            empirical_growths[inv].append(m)
    return {p: np.median(gs) for p, gs in empirical_growths.items()}


def calculate_intervention_behaviors(
        model, places, min_deaths, min_inv_days, max_pop_frac):
    growths = calculate_empirical_growths(
            places, min_deaths, min_inv_days, max_pop_frac)
    growths['Unknown'] = growths['No Intervention']
        # We conservatively treat the unknown intervention type as if it were no intervention.
    behaviors = {}
    for k, g in growths.items():
        b = Bunch()
        b.empirical_growth = g
        b.empirical_growth_beta = model.growth_rate_to_beta(g)
        b.ts = np.array([0], dtype=float)
        b.betas = np.array([b.empirical_growth_beta], dtype=float)
        behaviors[k] = b
    return behaviors


def calculate_death_trend(places, intervention):
    relative_deaths = collections.defaultdict(list)
    for p in places.values():
        if p.interventions is None: continue
        start_date = p.interventions.date_of_first(intervention)
        if start_date is None: continue
        if start_date not in p.deaths.dates(): continue
        start_idx = p.deaths.date_to_position(start_date)
        deaths_at_start = p.deaths[start_idx]
        if deaths_at_start < 5: continue
        for i, deaths in enumerate(p.deaths):
            time_delta = i-start_idx
            inv = p.interventions.extrapolate(p.deaths.date(i))
            if time_delta >= 0 and inv != intervention: break
            relative_deaths[time_delta].append(deaths/deaths_at_start)

    death_trend = []
    for day, rds in sorted(relative_deaths.items()):
        if day < 0: continue
        if len(rds) < 5: break
        death_trend.append(np.mean(rds))
    return death_trend


def fit_contact_rate_to_death_trend(model, trend, y0, keyframes, name=''):
    ts = np.arange(len(trend))
    n = len(keyframes)

    def curve_beta(params):
        beta = lambda t: np.interp(t, keyframes, params)
        setattr(beta, 'ts', keyframes)
        setattr(beta, 'betas', params)
        return beta

    def curve_fit_traj(params):
        with model.beta(curve_beta(params)):
            return model.integrate(y0, ts)

    def curve_fit(params):
        S, E, I, H, D, R = curve_fit_traj(params).T
        diff = np.linalg.norm(D - np.array(lockdown_death_trend, dtype=float))
        return diff

    starting_val = model.growth_rate_to_beta(1.2)
    curve_params = scipy.optimize.minimize(
            curve_fit, np.array([starting_val]*n), bounds=[(0, None)]*n).x
    for i,b in enumerate(curve_params):
        g = model.beta_to_growth_rate(b)
        print(f"\tLockdown: β_{i} = {b} ... growth={g}")

    if args.display_curve_fit:
        trajectories = curve_fit_traj(curve_params)
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, axisbelow=True)

        ax.set_title(f"Model Fit to {name} Death Trend")
        ax.set_xlabel(f'Days ({name} at 0)')
        ax.set_ylabel('People (log)')
        for var, curve in list(zip(Model.variables, trajectories.T))[1:]:
            ax.semilogy(ts, curve, label=var)
        ax.semilogy(ts, lockdown_death_trend, 's', label='D trend.')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        plt.show()
    return np.array(curve_params, dtype=float)


def interventions_to_beta_fn(intervention_behaviors, iv_seq, zero_day):
    margin = 0.01
    beta_starts = []
    prev_s = ''
    for d, s in iv_seq.items():
        if s != prev_s:
            beta_starts.append(((d-zero_day).days, intervention_behaviors[s]))
            prev_s = s
    ts = np.array([], dtype=float)
    betas = np.array([], dtype=float)
    for t, beh in beta_starts:
        if len(ts) == 0:
            ts = beh.ts+t
            betas = beh.betas.copy()
        else:
            t_end = t-margin
            beta_end = np.interp(t_end, ts, betas)
            ts = ts[ts < t_end]
            betas = betas[:len(ts)]
            ts = np.concatenate([ts, [t_end], beh.ts+t])
            betas = np.concatenate([betas, [beta_end], beh.betas])
    #print('----- created beta_fn:')
    #for t, beta in zip(ts, betas):
    #    d = (zero_day+datetime.timedelta(t)).isoformat()
    #    print(f"\tt={t} d={d} beta={beta}")
    #print('-----')
    fn = lambda t: np.interp(t, ts, betas)
    setattr(fn, 'ts', ts)
    setattr(fn, 'betas', betas)
    return fn


# --------------------------------------------------------------------------------
# Tuning procedure

# Set up a default version of the model:
model = Model(
        None,
        args.latent_period,
        args.infectious_period,
        args.p_hospital, args.hospital_duration,
        args.p_death_given_hospital)
print(f"Parameters: {model.param_str()}")


# Load the JHU time series data:
places = pickle.load(open('time_series.pkl', 'rb'))

intervention_behaviors = calculate_intervention_behaviors(
        model,
        places,
        args.empirical_growth_min_deaths,
        args.empirical_growth_min_inv_days,
        args.empirical_growth_max_pop_frac)

if args.optimize_lockdown:
    lockdown_death_trend = calculate_death_trend(places, 'Lockdown')
    with model.beta(intervention_behaviors['No Intervention'].empirical_growth_beta):
        y0 = model.equilibrium()[1]
    y0[0] = 1000000000
    keyframes = np.array([0, args.lockdown_warmup], dtype=float)
    betas = fit_contact_rate_to_death_trend(
            model, lockdown_death_trend, y0, keyframes, name="Lockdown")
    intervention_behaviors['Lockdown'].ts = keyframes
    intervention_behaviors['Lockdown'].betas = betas
    intervention_behaviors['Containment'].betas[0] = betas[1]/2.0
        # We assume that Containment is twice as effective in absolute terms as a hard lockdown.

for k, behavior in sorted(intervention_behaviors.items()):
    print(f"{k} ->")
    for t,beta in zip(behavior.ts, behavior.betas):
        print(f"\tβ({t})={beta}")



# --------------------------------------------------------------------------------
# Initialize Outputs:
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
        open('c19map.org_v2.0_Model_Data_-_Time_Series.csv', 'w'))
output_comprehensive_snapshot_w = csv.writer(
        open('c19map.org_v2.0_Model_Data_-_Snapshot.csv', 'w'))
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


if not args.nograph:
    graph_output_dir = 'graphs'
    if os.path.exists(graph_output_dir):
        shutil.rmtree(graph_output_dir)
    os.makedirs(graph_output_dir)

# Totals: for the historytable output.
history_dates = list(list(places.values())[0].deaths.dates())
world_confirmed = np.zeros(len(history_dates))
world_deaths = np.zeros(len(history_dates))
world_estimated_cases = np.zeros(len(history_dates))

# --------------------------------------------------------------------------------
# Run the model for each place.


for k, p in sorted(places.items()):
    N = p.population
    if N is None: continue
    if args.countries and k[0] not in args.countries: continue
    if args.places and ':'.join(k) not in args.places: continue

    present_date = p.deaths.last_date()
    print("Place =", p.region_id())

    # We find start_idx, the first death or first intervention.
    nz_deaths, = np.nonzero(p.deaths)
    if len(nz_deaths) == 0:
        print("No deaths recorded, skipping: ", p.region_id())
        continue
    first_death = nz_deaths[0]

    start_idx = first_death
    for i in range(first_death):
        if p.interventions.extrapolate(p.deaths.date(i)) != 'No Intervention':
            start_idx = i
            break

    # Next we simulate starting from one death at start_idx, and a very large population.
    with model.beta(intervention_behaviors['No Intervention'].empirical_growth_beta):
        growth_rate, equilibrium_state = model.equilibrium()
    fit_length = len(p.deaths)-start_idx
    beta = interventions_to_beta_fn(
            intervention_behaviors, p.interventions, present_date)
    t = np.arange(fit_length) - (fit_length+1)
    target = p.deaths[start_idx:]
    y0 = equilibrium_state.copy()
    large_pop = 10**10
    y0[0] = large_pop - np.sum(y0)
    with model.beta(beta):
        trajectories = model.integrate(y0, t)
    trajectories = trajectories[trajectories[:,0] > large_pop*0.9]
    #print("tj ->", type(trajectories))
      # Only keep the parts of the trajectory where less than 1/10 got infected.
    S, E, I, H, D, R = trajectories.T
    target = target[:len(D)]

    #print("D=",D)
    #print("D1=",type(D[1]))
    #print("target=",type(target[1]))

    # Then see how much to scale the death data to G
    def loss(s): return np.linalg.norm(D*s-target)
    state_scale = scipy.optimize.minimize_scalar(loss, bounds=(0.001, 1000)).x

    present_date = p.deaths.last_date()
    days_to_present = len(p.deaths) - 1 - start_idx
    days_simulation = days_to_present + args.graph_days_forecast + 1
    t = np.arange(-days_to_present, args.graph_days_forecast+1)
    
    y0 = state_scale * equilibrium_state
    y0[0] = N - np.sum(y0)

    with model.beta(beta):
        trajectories = model.integrate(y0, t)
        #trajectories_cython = model.integrate_cython(y0, t)

    def prepend_history(a, n, p, population):
        # Get the early history before start_idx by downscaling by the no-intervention growth rate.
        pre_history = np.outer(np.power(p, np.arange(-n,0)), a[0])
        # This works for all variables except S, so we have to fix S:
        pre_history[:,0] = population - pre_history[:,1:].sum(axis=1)
        return np.concatenate((pre_history, a))

    no_intervention_growth = intervention_behaviors['No Intervention'].empirical_growth
    trajectories = prepend_history(
            trajectories, start_idx, no_intervention_growth, N)
    #trajectories_cython = prepend_history(
    #        trajectories_cython, start_idx, no_intervention_growth, N)
    S, E, I, H, D, R = trajectories.T

    cumulative_infections = E+I+H+D+R  # Everyone who's ever been a case.
    active_infections = E+I+H

    # Update world history table:
    world_confirmed += p.confirmed
    world_deaths += p.deaths
    world_estimated_cases += cumulative_infections[:len(p.deaths)]

    # Output all variables:
    row_start = [k[0], k[1], p.latitude, p.longitude]
    all_vars_w.writerow(row_start + [round(x) for x in trajectories.T[:,len(p.deaths)]])

    # Output Estimation:
    latest_estimate = round(cumulative_infections[len(p.deaths)-1], -3)
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

    # CSV output helpers:
    def rnd(x, n=0): return np.around(x, n).astype(int)

    def per10k(x):
        return rnd(10000*x/p.population)

    def round_delta(x):
        return np.pad(rnd(x[1:]-x[:-1]), [(1, 0)], constant_values=0)

    def friendly_round_vec(x):
        return np.where(x>=1000, rnd(x, -3),
                np.where(x>=100, rnd(x, -2),
                    np.where(x>=10, rnd(x, -1),
                        np.zeros(x.shape, dtype=int))))

    jhu_fields = [p.confirmed.array(), p.deaths.array()]
    jhu_delta_fields = [round_delta(a) for a in jhu_fields]
    jhu_per10k_fields = [per10k(a) for a in jhu_fields]

    estimated_fields = list(trajectories.T) + [active_infections, cumulative_infections]
    estimated_per10k_fields = [per10k(a) for a in estimated_fields]
    estimated_delta_fields = [round_delta(a) for a in estimated_fields]
    estimated_fields = [friendly_round_vec(a) for a in estimated_fields]

    for idx, d in enumerate(p.deaths.dates()):
        intervention = p.interventions.extrapolate(d)
        initial_fields = [p.region_id(), d.isoformat(),
                p.display_name(), country, province,
                p.latitude, p.longitude,
                intervention, p.population]
        all_stat_fields = []
        for field_set in [jhu_fields, estimated_fields,
                jhu_delta_fields, estimated_delta_fields,
                jhu_per10k_fields, estimated_per10k_fields]:
            for field in field_set:
                if idx < len(field): all_stat_fields.append(field[idx])
                else: all_stat_fields.append(0)
        misc_fields = [present_date.isoformat(), '', '']
        all_fields = initial_fields + all_stat_fields + misc_fields
        output_comprehensive_series_w.writerow(all_fields)
        if d == present_date:
            output_comprehensive_snapshot_w.writerow(all_fields)

    # Graphs:
    if args.nograph: continue
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    intervention_starts = []
    old_s = 'No Intervention'
    for d,s in p.interventions.items():
        if s!=old_s: intervention_starts.append((d, s))
        old_s = s
    intervention_s = ', '.join(f"{s} on {d.strftime('%m-%d')}"
            for n,(d,s) in enumerate(intervention_starts))
    ax.set_title(p.region_id() + "\n" + intervention_s)
    ax.set_xlabel('Date')
    ax.xaxis.set_tick_params(which='both', labelsize=5, labelrotation=90)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=[1,15]))
    minor_locator = matplotlib.dates.DayLocator(bymonthday=[1]+list(range(5,26,5)))
    ax.xaxis.set_minor_locator(minor_locator)
    minor_formatter = matplotlib.dates.ConciseDateFormatter(minor_locator, show_offset=False)
    ax.xaxis.set_major_formatter(minor_formatter)
    ax.xaxis.set_minor_formatter(minor_formatter)
    ax.set_ylabel('People (log)')
    plt.grid(True)
    graph_dates = date_range_inclusive(
            p.deaths.start_date(),
            p.deaths.last_date() + datetime.timedelta(args.graph_days_forecast))
    if args.graph_back: s = 0
    else: s = first_death

    ax.axvline(present_date, 0, 1, linestyle='solid', color='black',
            label='Today')

    inv_colors = ["#9eb1bb", "#7d97a5", "#607d8b", "#57707d", "#4d636f"]
    for n, (inv_d, inv_str) in enumerate(intervention_starts):
        ax.axvline(inv_d, 0, 1, linestyle='dashed',
                color=inv_colors[n%len(inv_colors)],
                label=inv_str)
    for var, curve in zip(Model.variable_names, trajectories.T):
        ax.semilogy(graph_dates[s:], curve[s:], label=var, zorder=1)
    if 0:
        for var, curve in zip(Model.variable_names, trajectories_cython.T):
            ax.semilogy(graph_dates[s:], curve[s:], label="%"+var, linestyle='dashed', linewidth=5, zorder=2)
        for var, curve, curve_C in zip(Model.variable_names, trajectories.T, trajectories_cython.T):
            a = curve[len(p.deaths)-1]
            b = curve_C[len(p.deaths)-1]
            print(f"{var} -> off by {a/b} ({abs(a-b)})")

    ax.semilogy(graph_dates[s:len(p.deaths)], p.deaths[s:], 's',
            label='JHU deaths')
    ax.semilogy(graph_dates[s:len(p.deaths)], p.confirmed[s:], 's',
            label='JHU confirmed')
    if not args.graph_bottom: plt.ylim(bottom=1)
    ymin, ymax = ax.get_ylim()
    ax.yaxis.set_ticks([p for p in [10**n for n in range(12)] if p <= ymax])
    ax.yaxis.set_ticks([p for p in [a*10**n for n in range(12) for a in range(2,10)] if p <= ymax], minor=True)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    legend = ax.legend(bbox_to_anchor=(1.04,1), loc='upper left',
            fontsize='xx-small', borderaxespad=0)
    plt.savefig(os.path.join('graphs', p.region_id() + '.png'), dpi=100)
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
    estimated_cases = round(world_estimated_cases[i])
    history_w.writerow([short_date, date_str, confirmed, deaths, estimated_cases])
