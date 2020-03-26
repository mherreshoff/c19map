import numpy as np

class TimeSeries:
    def __init__(self, ts_len):
        self.latitude = None
        self.longitude = None
        self.confirmed = np.zeros(ts_len, dtype=int)
        self.deaths = np.zeros(ts_len, dtype=int)
        self.recovered = np.zeros(ts_len, dtype=int)

    def update(self, k, day, num):
        if k == 'confirmed': a = self.confirmed
        elif k == 'deaths': a = self.deaths
        elif k == 'recovered': a = self.recovered
        else: raise KeyError
        if num == '': return
        num = int(num)
        a[day] = max(a[day], num)
