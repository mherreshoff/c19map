import argparse
import csv
import datetime
import dateutil.parser
import numpy as np
import os

# Misc. Helpers:

def csv_as_matrix(path):
    return [r for r in csv.reader(open(path, 'r'))][1:]


def csv_as_dicts(source):
    csv_r = csv.reader(source)
    headers = next(csv_r)
    for row in csv_r:
        yield {col: x for col,x in zip(headers, row)}


def parse_date(s):
    if s == "today": return datetime.date.today()
    try: return dateutil.parser.parse(s).date()
    except ValueError: return None


def date_argument(s):
    d = parse_date(s)
    if d is None: raise argparse.ArgumentTypeError("Unparsable date: " + s)
    return d


def date_range_inclusive(start_date, end_date, delta=None):
    if delta is None: delta = datetime.timedelta(1)
    date = start_date
    while date <= end_date:
        yield date
        date += delta


def maybe_makedir(dirname):
    if not os.path.exists(dirname): os.makedirs(dirname)


class TimeSeries:
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

    def update(self, k, day, num):
        if k == 'confirmed': a = self.confirmed
        elif k == 'deaths': a = self.deaths
        elif k == 'recovered': a = self.recovered
        else: raise KeyError
        if num == '': return
        num = int(num)
        a[day] = max(a[day], num)




# Loading popultion data:
_population_data = None
def load_population_data():
    global _population_data
    if _population_data is None:
        _population_data = {
                (r[0], r[1], '') : int(r[3].replace(',',''))
                for r in csv_as_matrix('data_population.csv')}
    return _population_data



# Canonicalization/reconciliation:
recon_data_loaded = False
country_renames = None
place_renames = None
code_to_us_state = False
code_to_ca_province = False

# Figures out what places should be named.
def canonicalize_place(p):
    def sanetize(s):
        s = s.strip()
        if s == "None": s = ''
        return s
    global recon_data_loaded, country_renames, place_renames
    global code_to_us_state, code_to_ca_province
    if not recon_data_loaded:
        country_renames = {r[0]: r[1] for r in csv_as_matrix('data_country_renames.csv')}
        place_renames = {
                (r[0],r[1],r[2]): (r[3],r[4],r[5]) for r in csv_as_matrix('data_place_renames.csv')}
        code_to_us_state = {r[0]: r[3] for r in csv_as_matrix('data_us_states.csv')}
        code_to_ca_province = {r[0]: r[1] for r in csv_as_matrix('data_ca_provinces.csv')}
        recon_data_loaded = True
    p = tuple(map(sanetize, p))
    # Our models aren't about ships, so we ignore them:
    s = ';'.join(p)
    if 'Cruise Ship' in s or 'Princess' in s: return None

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
