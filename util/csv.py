import csv

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
