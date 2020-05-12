import os

from util.csv import csv_as_dicts

# Canonicalization/reconciliation:
class PlaceRecon:
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
