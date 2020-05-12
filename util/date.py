import argparse
import datetime
import dateutil.parser


def parse_date(s):
    """Parses dates.  Also accepts relative dates."""
    if s == "today": return datetime.date.today()
    elif s == "yesterday": return datetime.date.today() - datetime.timedelta(1)
    elif s == "tomorrow": return datetime.date.today() + datetime.timedelta(1)
    try: return dateutil.parser.parse(s).date()
    except ValueError: return None


def date_argument(s):
    """Use this for as type keyword of the ArgumentParser.add_argument method."""
    d = parse_date(s)
    if d is None: raise argparse.ArgumentTypeError("Unparsable date: " + s)
    return d


def date_range_inclusive(start_date, end_date, delta=None):
    """Produces a list of all dates from start_date to end_date (inclusive)."""
    r = []
    if delta is None: delta = datetime.timedelta(1)
    date = start_date
    while date <= end_date:
        r.append(date)
        date += delta
    return r
