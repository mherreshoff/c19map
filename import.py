#!/usr/bin/env python3
import argparse
import collections
import csv
import datetime
import io
import numpy as np
import os
import pickle
import urllib.request
import sys

from common import *

# --------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Make time series files from Johns Hopkins University data')
parser.add_argument("--start", default="2020-01-22", type=date_argument)
parser.add_argument("--last", default="today", type=date_argument)
parser.add_argument("--interventions_doc",
        default="1-tj7Cjx3e3eSfFGhhhJTGikaydIclFG1QrD06Eh9oDM")
parser.add_argument("--interventions_sheet", default="Interventions")
parser.add_argument("--population_doc",
        default="1-tj7Cjx3e3eSfFGhhhJTGikaydIclFG1QrD06Eh9oDM")
parser.add_argument("--population_sheet", default="population")
parser.add_argument("--sheets_csv_fetcher", default=(
    "https://docs.google.com/spreadsheets/d/{doc}/gviz/tq?tqx=out:csv&sheet={sheet}"))
parser.add_argument("--JHU_url_format", default=(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master'+
    '/csse_covid_19_data/csse_covid_19_daily_reports/%m-%d-%Y.csv'))
parser.add_argument('--JHU_data_dir', default='JHU_data')
args = parser.parse_args()


# --------------------------------------------------------------------------------
# Funtions which fetch data from JHU's github and our spreadsheets.

def fetch_raw_jhu_data(dates):
    jhu_data = {}
    for date in dates:
        url = date.strftime(args.JHU_url_format)
        file_path = os.path.join(args.JHU_data_dir, date.isoformat() + ".csv")
        if not os.path.exists(file_path):
            print(f"Downloading {file_path} from: {url}")
            try:
                urllib.request.urlretrieve(url, file_path)
            except urllib.error.HTTPError as e:
                print(f"Couldn't fetch: {url}")
                print(f"The fetch failed with code {e.code}: {e.reason}")
                if e.code == 404:
                    print(f"It seems JHU has not yet published the data for {date.isoformat()}.")
                sys.exit(1)
        jhu_data[date] = csv_as_dicts(open(file_path, encoding='utf-8-sig'))
        # Note: utf-8-sig gets rid of unicode byte order mark characters.
    return jhu_data


def fetch_population_data():
    """Download c19map.org's population data."""
    print("Downloading population data...")
    csv_url = args.sheets_csv_fetcher.format(
        doc=args.population_doc, sheet=args.population_sheet)
    csv_str = urllib.request.urlopen(csv_url).read().decode('utf-8')
    csv_source = csv_as_dicts(io.StringIO(csv_str))
    print("Done.")
    populations = {}
    for row in csv_source:
        country = row["Country/Region"]
        province = row["Province/State"]
        key = (country, province, '')
        populations[key] = int(row["Population"].replace(',', ''))
    return populations


def fetch_intervention_data():
    """Download c19map.org's intervention data."""
    print("Downloading intervention data...")
    csv_url = args.sheets_csv_fetcher.format(
        doc=args.interventions_doc, sheet=args.interventions_sheet)
    csv_str = urllib.request.urlopen(csv_url).read().decode('utf-8')
    csv_source = csv_as_dicts(io.StringIO(csv_str))
    print("Done.")
    date_cols = [(parse_date(s), s) for s in csv_source.headers()]
    date_cols = sorted((d, s) for d, s in date_cols if d is not None)
    for d, d2 in zip(date_cols, date_cols[1:]):
        assert (d2[0]-d[0]).days == 1, "Dates must be consecutive.  Did a column get deleted?"
    start_date = date_cols[0][0]
    unknown = TimeSeries(start_date, ['Unknown']*len(date_cols))

    interventions = {}
    for row in csv_source:
        place = (row['Country/Region'], row['Province/State'], '')
        assert place not in interventions, f"Duplicate row for place {place}"
        intervention_list = []
        prev_state = 'No Intervention'
        for d, s in date_cols:
            cell = row[s]
            if s == '':
                intervention_list.append(prev_state)
            else:
                intervention_list.append(cell)
                prev_state = cell
        interventions[place] = TimeSeries(start_date, intervention_list)
    return interventions, unknown

# --------------------------------------------------------------------------------
# Load our inputs:

dates = date_range_inclusive(args.start, args.last)
raw_jhu_data = fetch_raw_jhu_data(dates)
interventions, intervention_unknown = fetch_intervention_data()
intervention_dates = intervention_unknown.dates()
population = fetch_population_data()


# --------------------------------------------------------------------------------
# Reconcile the data together into one `Place` object for each region.
canonicalizer = PlaceCanonicalizer()
places = {}
interventions_recorded = set()
populations_recorded = set()
unknown_interventions_places = set()

throw_away_places = set([
    ('US', 'US', ''), ('Australia', '', ''),
    ('Canada', 'Recovered', ''),
    ('US', 'Recovered', ''),
    ])

def first_present(d, ks):
    for k in ks:
        if k in d: return d[k]
    return None

for date, row_source in raw_jhu_data.items():
    for keyed_row in row_source:
        country = first_present(keyed_row, ['Country_Region', 'Country/Region'])
        province = first_present(keyed_row, ['Province_State', 'Province/State'])
        district = first_present(keyed_row, ['Admin2']) or ''
        latitude = first_present(keyed_row, ['Lat', 'Latitude'])
        longitude = first_present(keyed_row, ['Long_', 'Longitude'])
        confirmed = first_present(keyed_row, ['Confirmed'])
        deaths = first_present(keyed_row, ['Deaths'])
        recovered = first_present(keyed_row, ['Recovered'])

        p = (country, province, district)
        if p in throw_away_places: continue
        p = canonicalizer.canonicalize(p)
        if p is None: continue

        if p not in places:
            places[p] = Place(dates)
            places[p].set_key(p)
            if p in population:
                places[p].population = population[p]
                populations_recorded.add(p)
            if p in interventions:
                places[p].interventions = interventions[p]
                interventions_recorded.add(p)
            else:
                places[p].interventions = intervention_unknown
                unknown_interventions_places.add(p)

        if latitude is not None and longitude is not None:
            places[p].latitude = latitude
            places[p].longitude = longitude
        places[p].update('confirmed', date, confirmed)
        places[p].update('deaths', date, deaths)
        places[p].update('recovered', date, recovered)


# These are the countries we already weren't bothering to simulate.
# By silencing them, these warnings can flag that a country got misspelt
# in or deleted from the population table (or JHU added a new country.)
SILENCE_POPULATION_WARNINGS = set([
    ('Australia', 'External territories', ''),
    ('Australia', 'Jervis Bay Territory', ''),
    ('France', 'Saint Pierre and Miquelon', ''),
    ('Netherlands', 'Bonaire, Sint Eustatius and Saba', ''),
    ('United Kingdom', 'Anguilla', ''),
    ('United Kingdom', 'British Virgin Islands', ''),
    ('United Kingdom', 'Falkland Islands (Islas Malvinas)', ''),
    ('United Kingdom', 'Falkland Islands (Malvinas)', ''),
    ('United Kingdom', 'Turks and Caicos Islands', ''),
    ('West Bank and Gaza', '', '')])

for p in sorted(places.values(), key=lambda p: p.key()):
    if p.population is None and p.district == '':
        if p.key() not in SILENCE_POPULATION_WARNINGS:
            print("No population data for ", p.region_id())

for p in sorted(interventions.keys()):
    if p not in interventions_recorded:
        print("Lost intervention data for: ", p)

# Consolidate US county data into states:
for p in sorted(places.keys()):
    if p[0] == "US" and p[2] != '':
        state = (p[0], p[1], '')
        if state not in places:
            places[state] = Place(dates)
            places[state].set_key(state)
            places[state].interventions = intervention_unknown
        places[state].confirmed += places[p].confirmed
        places[state].deaths += places[p].deaths
        places[state].recovered += places[p].recovered

# Fix the fact that France was recorded as French Polynesia on March 23rd:
france = places[('France', 'France', '')]
french_polynesia = places[('France', 'French Polynesia', '')]
idx = datetime.date(2020, 3, 23)
prev_idx = idx - datetime.timedelta(1)
france.confirmed[idx] = french_polynesia.confirmed[idx]
france.deaths[idx] = french_polynesia.deaths[idx]
france.recovered[idx] = french_polynesia.recovered[idx]
french_polynesia.confirmed[idx] = french_polynesia.confirmed[prev_idx]
french_polynesia.deaths[idx] = french_polynesia.deaths[prev_idx]
french_polynesia.recovered[idx] = french_polynesia.recovered[prev_idx]

# Hubei China suddenly increased their reported deaths on April 17th.
# Until we know what happened before then, we're scaling everything up:
hubei = places[('China', 'Hubei', '')]
idx = datetime.date(2020, 4, 17)
prev_idx = idx - datetime.timedelta(1)
scaling_factor = hubei.deaths[idx]/hubei.deaths[prev_idx]
pos = hubei.deaths.date_to_position(idx)
new_deaths = hubei.deaths.array()[:pos] * scaling_factor
hubei.deaths.array()[:pos] = new_deaths.astype(int)

# Output the CSVs:
confirmed_out = csv.writer(open("time_series_confirmed.csv", 'w'))
deaths_out = csv.writer(open("time_series_deaths.csv", 'w'))
recovered_out = csv.writer(open("time_series_recovered.csv", 'w'))

headers = ["Province/State","Country/Region","Lat","Long"]
headers += [d.strftime("%m/%d/%y") for d in dates]
confirmed_out.writerow(headers)
deaths_out.writerow(headers)
recovered_out.writerow(headers)

for p in sorted(places.keys()):
    country, province, district = p
    if district: continue
    if sum(places[p].confirmed) == 0: continue  #Skip if no data.

    latitude = places[p].latitude or ''
    longitude = places[p].longitude or ''
    row_start = [province, country, latitude, longitude]
    confirmed_out.writerow(row_start + list(places[p].confirmed))
    deaths_out.writerow(row_start + list(places[p].deaths))
    recovered_out.writerow(row_start + list(places[p].recovered))

# Output places dictionary:
timeseries_f = open('time_series.pkl', 'wb')
pickle.dump(places, timeseries_f)
timeseries_f.close()
