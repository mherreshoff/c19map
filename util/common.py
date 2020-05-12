import collections
import datetime
import dateutil.parser
import numpy as np
import numbers
import os


# -------------------------------------------------- 
# Generic helpers:

def constant_fn(val):
    def f(*args): return val
    return f

def maybe_makedir(dirname):
    if not os.path.exists(dirname): os.makedirs(dirname)


from util.time_series import TimeSeries

class Place:
    """All the data we know about a place:

    - Johns Hopkins time series for deaths, confirmed, and recovered
    - Population, latitude, and longitude.
    - Intervention time series.
    """
    def __init__(self, dates):
        self.country = None
        self.province = None
        self.district = None
        self.latitude = None
        self.longitude = None
        self.population = None

        self.confirmed = TimeSeries(dates[0], np.zeros(len(dates), dtype=int))
        self.deaths = TimeSeries(dates[0], np.zeros(len(dates), dtype=int))
        self.recovered =  TimeSeries(dates[0], np.zeros(len(dates), dtype=int))

        self.interventions = None

    def key(self):
        return (self.country, self.province, self.district)

    def set_key(self, k):
        self.country, self.province, self.district = k

    def region_id(self):
        return ' - '.join(k for k in self.key() if k != '')

    def display_name(self):
        parts = list(collections.OrderedDict.fromkeys(self.key()))
        return ' - '.join(p for p in parts if p != '')

    def update(self, k, day, num):
        if k == 'confirmed': a = self.confirmed
        elif k == 'deaths': a = self.deaths
        elif k == 'recovered': a = self.recovered
        else: raise KeyError
        if num == '': return
        num = int(num)
        a[day] = max(a[day], num)


from util.csv import csv_as_dicts

# Canonicalization/reconciliation:
class PlaceCanonicalizer:
    def __init__(self):
        def g(x): return csv_as_dicts(os.path.join('recon', x))
        self.country_renames = {r["Old Country"]: r["New Country"]
                for r in g('data_country_renames.csv')}
        self.place_renames = {}
        for r in g('data_place_renames.csv'):
            old = (r["Old Country"],r["Old Province"],r["Old District"])
            new = (r["New Country"],r["New Province"],r["New District"])
            self.place_renames[old] = new
        self.code_to_us_state = {r["state"]: r["name"]
                for r in g('data_us_states.csv')}
        self.code_to_ca_province = {r["Code"]: r["Province"]
                for r in g('data_ca_provinces.csv')}

    def sanetize(self, s):
        s = s.strip()
        if s == "None": s = ''
        return s

    def canonicalize(self, p):
        p = tuple(map(self.sanetize, p))
        # Our models aren't about ships, so we ignore them:
        for field in p:
            for ship_word in ['Cruise Ship', 'Princess', 'MS Zaandam']:
                if ship_word in field:
                    return None

        if p[0] in self.country_renames:
            p = (self.country_renames[p[0]], p[1], p[2])
        if p in self.place_renames: p = self.place_renames[p]

        if p[0] == "US":
            # Handle province fields like "Hubolt, CA"
            a = [x.strip() for x in p[1].split(',')]
            if len(a) == 2 and a[1] in self.code_to_us_state:
                p = (p[0], self.code_to_us_state[a[1]], a[0])
            # Remove the word 'County' from the district field.
            words = p[2].split(' ')
            if words[-1] == 'County':
                p = (p[0], p[1], ' '.join(words[:-1]))

        if p[0] == "Canada":
            # Handle province fields like "Montreal, QC"
            a = [x.strip() for x in p[1].split(',')]
            if len(a) == 2:
                province = self.code_to_ca_province.get(a[1], a[1])
                p = (p[0], province, a[0])
        return p
