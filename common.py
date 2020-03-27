import csv
import numpy as np


class TimeSeries:
    def __init__(self, dates):
        self.dates = dates
        self.latitude = None
        self.longitude = None
        self.confirmed = np.zeros(len(dates), dtype=int)
        self.deaths = np.zeros(len(dates), dtype=int)
        self.recovered = np.zeros(len(dates), dtype=int)

    def update(self, k, day, num):
        if k == 'confirmed': a = self.confirmed
        elif k == 'deaths': a = self.deaths
        elif k == 'recovered': a = self.recovered
        else: raise KeyError
        if num == '': return
        num = int(num)
        a[day] = max(a[day], num)


def csv_as_matrix(path):
    return [r for r in csv.reader(open(path, 'r'))][1:]


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
