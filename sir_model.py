#!/usr/bin/env python3
import csv
import datetime
import pickle

from common import *

places = pickle.load(open('time_series.pkl', 'rb'))

def csv_as_matrix(path):
    return [r for r in csv.reader(open(path, 'r'))][1:]

def parse_int(s):
    return int(s.replace(',', ''))

population = {(r[0], r[1], '') : parse_int(r[3])
        for r in csv_as_matrix('data_population.csv')}

for k in sorted(population.keys()):
    if k in places:
        print(k)
