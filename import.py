#!/usr/bin/env python3
import argparse
import collections
import csv
import datetime
import dateutil.parser
import io
import numpy as np
import os
import pickle
import urllib.request

from common import *


parser = argparse.ArgumentParser(description='Make time series files from Johns Hopkins University data')
parser.add_argument("-s", "--start", default="2020-01-22")
parser.add_argument("-e", "--end", default="today")
parser.add_argument("--interventions_doc", default="1Rl3uhYkKfZiYiiRyJEl7R5Xay2HNT20R1X1j1nDCnd8")
parser.add_argument("--interventions_sheet", default="Interventions")
parser.add_argument("--sheets_csv_fetcher", default=(
    "https://docs.google.com/spreadsheets/d/{doc}/gviz/tq?tqx=out:csv&sheet={sheet}"))
parser.add_argument("--JHU_url_format", default=(
   'https://raw.githack.com/CSSEGISandData/COVID-19/master'+
   '/csse_covid_19_data/csse_covid_19_daily_reports/%m-%d-%Y.csv'))
data_directory = 'JHU_data'
args = parser.parse_args()

start_date = datetime.date.fromisoformat(args.start)
if args.end == "today": end_date = datetime.date.today()
else: end_date = datetime.date.fromisoformat(args.end)


# Download Johns Hopkins Data:
downloads = []
day_count = (end_date - start_date).days + 1;
dates = []

if not os.path.exists(data_directory): os.makedirs(data_directory)

for n in range(day_count):
    d = start_date + datetime.timedelta(n)
    dates.append(d)
    url = d.strftime(args.JHU_url_format)
    file_path = os.path.join(data_directory, d.isoformat() + ".csv")
    downloads.append([url, file_path, n])

for url, file_path, day in downloads:
    if not os.path.exists(file_path):
        print("Downloading "+file_path+" from: "+url)
        urllib.request.urlretrieve(url, file_path)


def fetch_intervention_data():
    """Download the c19map.org's intervention data."""
    csv_url = args.sheets_csv_fetcher.format(
        doc=args.interventions_doc, sheet=args.interventions_sheet)
    csv_str = urllib.request.urlopen(csv_url).read().decode('utf-8')
    intervention_csv = csv.reader(io.StringIO(csv_str))

    headers = next(intervention_csv)
    province_col = headers.index("Province/State")
    country_col = headers.index("Country/Region")
    date_cols = []
    dates = []
    for i, s in enumerate(headers):
        try:
            d = dateutil.parser.parse(s).date()
            date_cols.append(i)
            dates.append(d)
        except ValueError:
            pass # Not a date column
    for d, d2 in zip(dates, dates[1:]):
        assert (d2-d).days == 1, "Dates must be consecutive.  Did a column get deleted?"

    interventions = {}
    for row in intervention_csv:
        country = row[country_col]
        province = row[province_col]
        old_p = (country, province, '')
        p = canonicalize_place(old_p)
        if p != old_p:
            print("Non-canonical place: ",p," detected in interventions.")
        if p in interventions:
            print("Duplicate rows for place ",p)
        ivs = [row[c] for c in date_cols]
        interventions[p] = (dates, ivs)
    return interventions


print("Fetching latest intervention data")
interventions = fetch_intervention_data()


# Read our JHU data, and reconcile it together:
places = {}
interventions_recorded = set()
throw_away_places = set([('US', 'US', ''), ('Australia', '', '')])

def first_of(d, ks):
    for k in ks:
        if k in d: return d[k]
    return None

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
        if p in throw_away_places: continue
        p = canonicalize_place(p)
        if p is None: continue

        if p not in places:
            places[p] = TimeSeries(dates)
            if p in interventions:
                iv_ds, ivs = interventions[p]
                places[p].intervention_dates = iv_ds
                places[p].interventions = ivs
                interventions_recorded.add(p)

        if latitude is not None and longitude is not None:
            places[p].latitude = latitude
            places[p].longitude = longitude
        places[p].update('confirmed', day, confirmed)
        places[p].update('deaths', day, deaths)
        places[p].update('recovered', day, recovered)

for p in interventions.keys():
    if p not in interventions_recorded:
        print("Lost intervention data for: ", p)

# Consolidate US county data into states:
for p in sorted(places.keys()):
    if p[0] == "US" and p[2] != '':
        state = (p[0], p[1], '')
        if state not in places: places[state] = TimeSeries(dates)
        places[state].confirmed += places[p].confirmed
        places[state].deaths += places[p].deaths
        places[state].recovered += places[p].recovered

# Merge in French Polynesia:
france = ('France', 'France', '')
french_polynesia = ('France', 'French Polynesia', '')
if french_polynesia in places:
    places[france].confirmed += places[french_polynesia].confirmed
    places[france].deaths += places[french_polynesia].deaths
    places[france].recovered += places[french_polynesia].recovered
    del places[french_polynesia]



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
