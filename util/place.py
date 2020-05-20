import collections
import numpy as np

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
        self.google_mobility = None

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
