import argparse
import collections
import csv
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


def friendly_round(n):
    if not isinstance(n, numbers.Number): return n
    if n >= 1000: return round(n, -3)
    if n >= 100:  return round(n, -2)
    if n >= 10:   return round(n, -1)
    return 0


class csv_as_dicts:
    def __init__(self, source):
        if isinstance(source, str): source = open(source, 'r')
        self._csv_reader = csv.reader(source)
        self._headers = next(self._csv_reader)

    def headers(self):
        return self._headers

    def set_headers(self, headers):
        assert len(headers) == len(self._headers), "Can't change column count."
        self._headers = headers

    def __iter__(self):
        for row in self._csv_reader:
            yield {h: x for h,x in zip(self._headers, row)}


def parse_date(s):
    if s == "today": return datetime.date.today()
    elif s == "yesterday": return datetime.date.today() - datetime.timedelta(1)
    elif s == "tomorrow": return datetime.date.today() + datetime.timedelta(1)
    try: return dateutil.parser.parse(s).date()
    except ValueError: return None


def date_argument(s):
    d = parse_date(s)
    if d is None: raise argparse.ArgumentTypeError("Unparsable date: " + s)
    return d


def date_range_inclusive(start_date, end_date, delta=None):
    r = []
    if delta is None: delta = datetime.timedelta(1)
    date = start_date
    while date <= end_date:
        r.append(date)
        date += delta
    return r


def maybe_makedir(dirname):
    if not os.path.exists(dirname): os.makedirs(dirname)


class TimeSeries:
    """Represents a time series of data using any array-like object and
    a start date."""
    def __init__(self, start_date, array):
        self._start_date = start_date
        self._array = array

    def array(self):
        return self._array

    def start_date(self):
        return self._start_date

    def date(self, n):
        return self._start_date + datetime.timedelta(n)

    def last_date(self):
        return self.date(len(self._array)-1)

    def stop_date(self):
        return self.date(len(self._array))

    def dates(self):
        for i, _ in enumerate(self._array):
            yield self.date(i)

    def date_of_first(self, x):
        try: return self.date(self._array.index(x))
        except ValueError: return None

    def date_to_position(self, date, extrapolate=False, cutoff=True):
        n = (date - self._start_date).days
        if n < 0:
            if extrapolate: return 0
            elif cutoff: return None
            else: return n
        if n >= len(self._array):
            if extrapolate: return len(self._array)-1
            elif cutoff: return None
            else: return n
        return n

    def index_to_position(self, idx, extrapolate=False, cutoff=True):
        if isinstance(idx, datetime.date):
            n = self.date_to_position(idx, extrapolate=extrapolate, cutoff=cutoff)
        elif isinstance(idx, slice):
            start = self.index_to_position(idx.start, extrapolate=False, cutoff=False)
            stop = self.index_to_position(idx.stop, extrapolate=False, cutoff=False)
            step = idx.step  # support timedelta here?
            n = slice(start, stop, step)
        else:
            n = idx
        return n

    def extrapolate(self, date):
        n = self.date_to_position(date, extrapolate=True)
        return self._array[n]

    def __len__(self):
        return len(self._array)

    def __getitem__(self, idx):
        n = self.index_to_position(idx)
        return self._array[n]

    def __setitem__(self, idx, val):
        n = self.index_to_position(idx)
        self._array[n] = val

    def __iter__(self):
        return iter(self._array)

    def __add__(self, other):
        assert self._start_date == other._start_date
        assert len(self._array) == len(other._array)
        return TimeSeries(self._start_date, self._array + other._array)

    def items(self):
        for i, x in enumerate(self._array):
            yield (self.date(i), x)



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
