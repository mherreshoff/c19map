#!/usr/bin/env python3
import csv
import datetime
import collections
import urllib.request
import os

start_date = datetime.date(2020, 1, 22)
end_date = datetime.date(2020, 3, 23)
url_prefix = 'https://raw.githack.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'

# Import Data Files:
country_renames_rows = [r for r in csv.reader(open('data_country_renames.csv', 'r'))]
country_renames = {r[0]: r[1] for r in country_renames_rows[1:]}

place_renames_rows = [r for r in csv.reader(open('data_place_renames.csv', 'r'))]
place_renames = {
        (r[0],r[1],r[2]): (r[3], r[4], r[5]) for r in place_renames_rows[1:]}

canonical_regions = [(r[0], r[1]) for r in csv.reader(open('data_canonical_regions.csv', 'r'))][1:]

us_states_rows = [r for r in csv.reader(open('data_us_states.csv', 'r'))]
code_to_us_state = {r[0]: r[3] for r in us_states_rows[1:]}
code_to_ca_province = {
    'AB': 'Alberta', 'BC': 'British Columbia', 'MB': 'Manitoba',
    'NB': 'New Brunswick', 'NL': 'Newfoundland and Labrador',
    'NT': 'Northwest Territories', 'NS': 'Nova Scotia',
    'NU': 'Nunavut', 'ON': 'Ontario', 'PE': 'Prince Edward Island',
    'QC': 'Quebec', 'SK': 'Saskatchewan', 'YT': 'Yukon'}

# Figures out what places should be named.
def canonicalize_place(p):
    if p[0] in country_renames:
        p = (country_renames[p[0]], p[1], p[2])
    if p in place_renames: p = place_renames[p]

    if (p[0], p[1]) not in canonical_regions:
        # If this region isn't canonical, let's try to canonicalize it.
        if (p[1], p[2]) in canonical_regions: p = (p[1], p[2], '')
            # E.g. If they put Taiwan inside of China, and we don't, we extract it.
        else:
            for r in canonical_regions:
                if r[1] == p[0]:
                    p = (r[0], p[0], p[1])
                    break
                # E.g. If we consider Greenland part of Denmark and they don't we put it back in.
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
    if p in place_renames: p = place_renames[p]
    return p



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
        district = sanetize(first_of(keyed_row, ['Admin2']) or '')
        latitude = first_of(keyed_row, ['Lat', 'Latitude'])
        longitude = first_of(keyed_row, ['Long_', 'Longitude'])
        confirmed = first_of(keyed_row, ['Confirmed'])
        deaths = first_of(keyed_row, ['Deaths'])
        recovered = first_of(keyed_row, ['Recovered'])

        place = (country, province, district)
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
    country, province, district = place
    if district:
        division = province + " - " + district
    else:
        division = province

    latitude = latitude_by_place.get(place, '')
    longitude = longitude_by_place.get(place, '')
    row_start = [country, division, latitude, longitude]
    confirmed_out.writerow(row_start + confirmed_by_place[place])
    deaths_out.writerow(row_start + deaths_by_place[place])
    recovered_out.writerow(row_start + recovered_by_place[place])

