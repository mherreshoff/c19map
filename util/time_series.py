import datetime

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

    # TODO: come up with more intuitive ways for these arguments to work.
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
