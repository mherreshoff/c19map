#!/usr/bin/env python3
import argparse
import collections
import csv
import datetime
import numpy as np
import os
import pickle
import urllib.request

from common import *


parser = argparse.ArgumentParser(description='Make time series files from Johns Hopkins University data')
parser.add_argument("-s", "--start", default="2020-01-22")
parser.add_argument("-e", "--end", default="today")
url_prefix = 'https://raw.githack.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'
data_directory = 'JHU_data'
args = parser.parse_args()

start_date = datetime.date.fromisoformat(args.start)
if args.end == "today":
    end_date = datetime.date.today()
else:
    end_date = datetime.date.fromisoformat(args.end)

# Import data:
def csv_as_matrix(path):
    return [r for r in csv.reader(open(path, 'r'))][1:]

code_to_us_state = {r[0]: r[3] for r in csv_as_matrix('data_us_states.csv')}
code_to_ca_province = {r[0]: r[1] for r in csv_as_matrix('data_ca_provinces.csv')}

country_renames = {r[0]: r[1] for r in csv_as_matrix('data_country_renames.csv')}
place_renames = {
        (r[0],r[1],r[2]): (r[3],r[4],r[5]) for r in csv_as_matrix('data_place_renames.csv')}

throw_away_places = set([('US', 'US', ''), ('Australia', '', '')])

# Download Johns Hopkins Data:
downloads = []
day_count = (end_date - start_date).days + 1;
dates = []

if not os.path.exists(data_directory): os.makedirs(data_directory)

for n in range(day_count):
    d = start_date + datetime.timedelta(n)
    dates.append(d)
    file_name = d.strftime('%m-%d-%Y.csv')
    file_path = os.path.join(data_directory, file_name)
    downloads.append([url_prefix + file_name, file_path, n])

for url, file_path, day in downloads:
    if not os.path.exists(file_path):
        print("Downloading "+file_path+" url: "+url)
        urllib.request.urlretrieve(url, file_path)


def sanetize(s):
    s = s.strip()
    if s == "None": s = ''
    return s


# Figures out what places should be named.
def canonicalize_place(p):
    p = tuple(map(sanetize, p))
    # Our models aren't about ships, so we ignore them:
    s = ';'.join(p)
    if 'Cruise Ship' in s or 'Princess' in s: return None
    if p in throw_away_places: return None

    if p[0] in country_renames:
        p = (country_renames[p[0]], p[1], p[2])
    if p in place_renames: p = place_renames[p]

    if p[0] == "US":
        # Handle province fields like "Hubolt, CA"
        a = p[1].split(',')
        if len(a) == 2:
            state = code_to_us_state.get(a[1].strip(), None)
            if state:
                p = (p[0], state, a[0].strip())
        # Remove the word 'County' from the district field.
        words = p[2].split(' ')
        if words[-1] == 'County':
            p = (p[0], p[1], ' '.join(words[:-1]))
    if p[0] == "Canada":
        # Handle province fields like "Montreal, QC"
        a = p[1].split(',')
        if len(a) == 2:
            a[0] = a[0].strip()
            a[1] = a[1].strip()
            province = code_to_ca_province.get(a[1], a[1])
            p = (p[0], province, a[0])
    return p


def first_of(d, ks):
    for k in ks:
        if k in d:
            return d[k]
    return None



# Read our JHU data, and reconsile it together:
places = {}

for url, file_name, day in downloads:
    rows = [row for row in csv.reader(open(file_name,encoding='utf-8-sig'))]
    for i in range(1, len(rows)):
        keyed_row = dict(zip(rows[0], rows[i]))

        country = first_of(keyed_row, ['Country_Region', 'Country/Region'])
        province = first_of(keyed_row, ['Province_State', 'Province/State'])
        district = first_of(keyed_row, ['Admin2']) or ''
        latitude = first_of(keyed_row, ['Lat', 'Latitude'])
        longitude = first_of(keyed_row, ['Long_', 'Longitude'])
        confirmed = first_of(keyed_row, ['Confirmed'])
        deaths = first_of(keyed_row, ['Deaths'])
        recovered = first_of(keyed_row, ['Recovered'])

        p = (country, province, district)
        p = canonicalize_place(p)
        if p is None: continue

        if p not in places: places[p] = TimeSeries(day_count)

        if latitude is not None and longitude is not None:
            places[p].latitude = latitude
            places[p].longitude = longitude
        places[p].update('confirmed', day, confirmed)
        places[p].update('deaths', day, deaths)
        places[p].update('recovered', day, recovered)


# Consolidate US county data into states:
for p in sorted(places.keys()):
    if p[0] == "US" and p[2] != '':
        state = (p[0], p[1], '')
        if state not in places: places[state] = TimeSeries(day_count)
        places[state].confirmed += places[p].confirmed
        places[state].deaths += places[p].deaths
        places[state].recovered += places[p].recovered


# Output the CSVs:
confirmed_out = csv.writer(open("time_series_confirmed.csv", 'w'))
deaths_out = csv.writer(open("time_series_deaths.csv", 'w'))
recovered_out = csv.writer(open("time_series_recovered.csv", 'w'))

headers = ["Province/State","Country/Region","Lat","Long"]
headers += ["%d/%d/%d" % (d.month, d.day, d.year%100) for d in dates]
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
