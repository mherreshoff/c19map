import country_converter as coco
import logging
import os

from util.csv import csv_as_dicts

seen = set()

# Canonicalization/reconciliation:
class PlaceRecon:
    def __init__(self):
        logging.disable(logging.WARNING)
            # country_converter issues warnings if you try to convert something that isn't
            # a country, but I want to use it to test for countries.
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

        self.country_cache = {}
        self.place_cache = {}
        self.used_renames = set()
        self.unrecognized_countries = set()

    def sanetize(self, s):
        s = s.strip()
        if s == "None": s = ''
        return s

    def canonicalize_country(self, c):
        if c in self.country_cache: return self.country_cache[c]
        name = coco.convert(names=c, src='regex', to='short_name', not_found='??')
        if name == '??':
            self.unrecognized_countries.add(c)
            if c == "Republic of Ireland":
                name = "Ireland"
            else:
                name = c
        self.country_cache[c] = name
        return name

    def canonicalize(self, p):
        if p in self.place_cache: return self.place_cache[p]
        old_p = p
        p = tuple(map(self.sanetize, p))
        # Our models aren't about ships, so we ignore them:
        for field in p:
            for ship_word in ['Cruise Ship', 'Princess', 'MS Zaandam']:
                if ship_word in field:
                    self.place_cache[old_p] = None
                    return None

        # Sometimes the old country name gets duplicated.
        while p[0] and p[0] == p[1]:
            p = (p[0], p[2], '')

        country_name = self.canonicalize_country(p[0])
        p = (country_name, p[1], p[2])

        # Sometimes the new country name gets duplicated.
        while p[0] and p[0] == p[1]:
            p = (p[0], p[2], '')

        if p[1] == p[2]:
            p = (p[0], p[1], '')

        if p in self.place_renames:
            self.used_renames.add(p)
            p = self.place_renames[p]

        if p[0] == "United States":
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
        self.place_cache[old_p] = p
        if p[0] != 'United States' and p != old_p:
            print(f'canonicalized\n\t{old_p} --->\n\t{p}\n')
        return p
