#!/usr/bin/env python3
import csv
import datetime
import collections
import urllib.request
import os

start_date = datetime.date(2020, 1, 22)
end_date = datetime.date(2020, 3, 23)
url_prefix = 'https://raw.githack.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'

country_renames = {
    'Mainland China': 'China',
    'Bahamas, The': 'Bahamas',
    'Gambia, The': 'Gambia',
    'Czech Republic': 'Czechia',
    'Viet Nam': 'Vietnam',
    'Vatican City': 'Holy See',
    'Taiwan': 'Taiwan*',
    'East Timor': 'Timor-Leste',
}

canonical_regions = [(r[0], r[1]) for r in csv.reader(open('data_regions.csv', 'r'))][1:]

us_states_rows = [r for r in csv.reader(open('data_us_states.csv', 'r'))]
code_to_us_state = {r[0]: r[3] for r in us_states_rows[1:]}

def canonicalize_place(p):
    if (p[0], p[1]) in canonical_regions: return p
        # If this region is canonical, we're done.
    if (p[1], p[2]) in canonical_regions: return (p[1], p[2], '')
        # E.g. If they accidentaly put Taiwan inside of China, we extract it.
    for r in canonical_regions:
        if r[1] == p[0]:
            return (r[0], p[0], p[1])
        # E.g. If we consider Greenland part of Denmark and they don't we put it back in.
    if p[0] == "US":
        a = p[1].split(',')
        if len(a) == 2:
            state = code_to_us_state.get(a[1].strip(), None)
            if state:
                return (p[0], state, a[0].strip())
    return p
        # If all of those attempts fail, let it stand.



downloads = []
day_count = (end_date - start_date).days + 1;
dates = []
for n in range(day_count):
    d = start_date + datetime.timedelta(n)
    dates.append(d)
    file_name = d.strftime('%m-%d-%Y.csv')
    downloads.append([url_prefix + file_name, file_name, n])


def first_of(d, ks):
    for k in ks:
        if k in d:
            return d[k]
    return None

def sanetize(s):
    s = s.strip()
    if s == "None": s = ''
    return s

places = set()
latitude_by_place = {} 
longitude_by_place = {} 
confirmed_by_place = collections.defaultdict(lambda:['']*day_count)
deaths_by_place = collections.defaultdict(lambda:['']*day_count)
recovered_by_place = collections.defaultdict(lambda:['']*day_count)

for url, file_name, day in downloads:
    if not os.path.exists(file_name):
        print("Downloading "+file_name+"...");
        urllib.request.urlretrieve(url, file_name)
    rows = [row for row in csv.reader(open(file_name,encoding='utf-8-sig'))]
    for i in range(1, len(rows)):
        keyed_row = dict(zip(rows[0], rows[i]))

        country = sanetize(first_of(keyed_row, ['Country_Region', 'Country/Region']))
        province = sanetize(first_of(keyed_row, ['Province_State', 'Province/State']))
        county = sanetize(first_of(keyed_row, ['Admin2']) or '')
        latitude = first_of(keyed_row, ['Lat', 'Latitude'])
        longitude = first_of(keyed_row, ['Long_', 'Longitude'])
        confirmed = first_of(keyed_row, ['Confirmed'])
        deaths = first_of(keyed_row, ['Deaths'])
        recovered = first_of(keyed_row, ['Recovered'])

        if country in country_renames:
            country = country_renames[country]

        place = (country, province, county)
        place = canonicalize_place(place)

        if place not in places:
            places.add(place)
        if latitude is not None and longitude is not None:
            latitude_by_place[place] = latitude
            longitude_by_place[place] = longitude
        confirmed_by_place[place][day] = confirmed
        deaths_by_place[place][day] = deaths
        recovered_by_place[place][day] = recovered


# Output the CSVs:
confirmed_out = csv.writer(open("output_confirmed_data.csv", 'w'))
deaths_out = csv.writer(open("output_deaths_data.csv", 'w'))
recovered_out = csv.writer(open("output_recovered_data.csv", 'w'))

headers = ["Province/State","Country/Region","Lat","Long"]
headers += ["%d/%d/%d" % (d.month, d.day, d.year%100) for d in dates]
confirmed_out.writerow(headers)
deaths_out.writerow(headers)
recovered_out.writerow(headers)

for place in sorted(places):
    country, province, county = place
    if ',' in province: print(place)

for place in sorted(places):
    country, province, county = place
    if county:
        division = province + " - " + county
    else:
        division = province
    if deaths_by_place[place][-1] == '':
        print("D", place)

    latitude = latitude_by_place.get(place, '')
    longitude = longitude_by_place.get(place, '')
    row_start = [country, division, latitude, longitude]
    confirmed_out.writerow(row_start + confirmed_by_place[place])
    deaths_out.writerow(row_start + deaths_by_place[place])
    recovered_out.writerow(row_start + recovered_by_place[place])

