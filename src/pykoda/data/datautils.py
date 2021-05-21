import os
import glob
import inspect
import warnings
import datetime
import functools
import contextlib
import collections

import numpy as np
import pandas as pd
import tqdm

from . import getdata
from . import getstatic
from .. import config


def _filter_df(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if '@' in query:
        # In this case, we appear to be using
        # external variables, so we need to
        # pull in the local variables from the caller

        frame = inspect.currentframe()
        caller_locals = frame.f_back.f_locals
        with contextlib.suppress(TypeError):
            # We hop back two frames, so the caller and the caller's caller.
            caller_locals.update(frame.f_back.f_back.f_locals)

    else:
        # Otherwise, leave it empty.
        caller_locals = None

    try:
        return df.query(query, local_dict=caller_locals)
    except TypeError:
        # This may be triggered by an operation unsupported by Numexpr.
        # Try to fall back to Python backend.
        return df.query(query, local_dict=caller_locals, engine='python')
    except pd.core.computation.ops.UndefinedVariableError:
        msg = ('One of the keys does not seem to appear in some of the data. '
               'This may be due to wrong spelling, or simply missing from one of the hours.')
        warnings.warn(RuntimeWarning(msg))
        return pd.DataFrame()


def _clear_duplicates(df: pd.DataFrame, feed: str) -> None:
    if feed == 'TripUpdates' and not df.empty:
        df.sort_values(by='timestamp', ascending=True)

        # Not all feeds provide exactly the same fields, so this filters for it:
        keys = list({'trip_id', 'direction_id', 'stop_sequence', 'stop_id'}.intersection(df.keys()))
        df.drop_duplicates(subset=keys, keep='last', inplace=True)


def get_data_range(feed: str, company: str, start_date: str, start_hour: int = 0, end_date: str = None,
                   end_hour: int = 23, query: (str, None) = None, merge_static: bool = False,
                   clear_duplicates: bool = True) -> pd.DataFrame:
    """Get all data from start_date:start_hour to end_date:end_hour, inclusive, downloading if necessary.
    Date should be given in the format YYYY-MM-DD or YYYY_MM_DD.

    If end_date is not specified, it is the same as start_date.

    Use the parameter `query` to filter the data as it is being read if you want to minimise the memory footprint.

    merge_static includes some information from the static feed.
    """
    if feed not in ('VehiclePositions', 'TripUpdates', 'ServiceAlerts'):
        raise ValueError(f'Feed {feed} not recognised.')
    if company not in ('dt', 'klt', 'otraf', 'skane', 'sl', 'ul', 'varm', 'xt'):
        warnings.warn(RuntimeWarning(f'Company {company} is not recognised. Maybe the API will fail?'))

    frames = []
    if end_date is None:
        end_date = start_date

    start = start_date.replace('_', '-') + f' {start_hour}:00'
    end = end_date.replace('_', '-') + f' {end_hour}:00'

    for date_hour in tqdm.tqdm(pd.date_range(start, end, freq='h'), desc='Loading data'):
        date, time_code = str(date_hour).split()
        hour = int(time_code.split(':')[0])
        full_path = getdata._get_data_path(company, feed=feed, hour=hour, date=date)
        if not os.path.exists(full_path):
            # Download if it doesn't exist'
            try:
                getdata.get_data(date, hour, feed, company)
            except ValueError:
                warnings.warn(RuntimeWarning(f'The API did not return data for {date_hour}'))
                continue

        df = pd.read_feather(full_path)
        if clear_duplicates:
            _clear_duplicates(df, feed)

        if feed == 'VehiclePositions':
            if 'trip_id' in df:
                # For vehicle positions, skip data that does not correspond to routes.
                df = df.query('trip_id.notna()', engine='python')
            else:
                df = pd.DataFrame()

        if merge_static and not df.empty:
            this_static = load_static_data(company, date, remove_unused_stations=True)
            if this_static is None:
                continue

            if feed == 'TripUpdates':
                # Overwrite some parameters already present in the static data
                df.drop(columns=['route_id', 'direction_id'], errors='ignore', inplace=True)

                # Rename some common columns
                df.rename(
                    columns={'arrival_time': 'observed_arrival_time', 'departure_time': 'observed_departure_time'},
                    inplace=True)

                this_static.stop_times.rename(columns={'arrival_time': 'scheduled_arrival_time',
                                                       'departure_time': 'scheduled_departure_time'}, inplace=True)

                df = df.merge(this_static.stop_times, how='left',
                              on=('trip_id', 'stop_id', 'stop_sequence'),
                              validate='m:1')
                df = df.merge(this_static.trips.reset_index(level=['route_id', 'direction_id']), how='left',
                              on='trip_id', validate='m:1')

            elif feed == 'VehiclePositions':
                df.drop(columns=['route_id', 'direction_id'], errors='ignore', inplace=True)
                df = df.merge(this_static.trips.reset_index(level=['route_id', 'direction_id']), how='left',
                              on='trip_id', validate='m:1')

            elif feed == 'ServiceAlerts':
                warnings.warn(RuntimeWarning('ServiceAlerts cannot be merged with static data.'))

        if query is not None:
            df = _filter_df(df, query)

        # Drop the index, since it will be regenerated when concatenated
        df.drop(columns='index', errors='ignore', inplace=True)
        frames.append(df)

    df_merged = pd.concat(frames)
    if 'timestamp' in df_merged.keys():
        df_merged['datetime'] = pd.to_datetime(df_merged.timestamp, unit='s')
    return df_merged


static_data = collections.namedtuple('StaticData', ['stop_times', 'stops', 'trips', 'shapes', 'routes'])


def _remove_unused_stations(stops_data: pd.DataFrame, stops_times: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    parent_stations = stops_data.parent_station.dropna().values
    used_stations = set(stops_times.stop_id).union(parent_stations)
    stops_times = stops_times.query('stop_id in @used_stations')
    stops_data = stops_data.query('stop_id in @used_stations')
    return stops_data, stops_times


@functools.lru_cache(1)
def load_static_data(company: str, date: str, remove_unused_stations: bool = False) -> (static_data, None):
    """Load static data from the cache, downloading it if necessary.

    Date should be given in the format YYYY-MM-DD or YYYY_MM_DD.
    """
    output_folder = getstatic._get_static_data_path(company, date)
    if not os.path.isdir(output_folder):
        try:
            getstatic.get_static_data(company=company, date=date)
        except ValueError:
            warnings.warn(RuntimeWarning(f'Missing static data for {company} at {date}'))
            return

    if not os.path.exists(os.path.join(output_folder, 'stop_times.txt')):
        warnings.warn(RuntimeWarning(f'Missing static data for {company} at {date}'))
        return

    def parse_date(time_code: str):
        time_code = list(map(int, time_code.split(':')))
        if time_code[0] < 24:
            return datetime.datetime(*map(int, date.replace('_', '-').split('-')), *time_code)
        else:
            # If it is over 24 h, roll over the next day.
            time_code[0] -= 24
            dt = datetime.datetime(*map(int, date.replace('_', '-').split('-')), *time_code)
            return dt + datetime.timedelta(days=1)

    stops_times = pd.read_csv(os.path.join(output_folder, 'stop_times.txt'), dtype={'trip_id': 'str', 'stop_id': 'str'},
                              parse_dates=['arrival_time', 'departure_time'], date_parser=parse_date)
    stops_data = pd.read_csv(os.path.join(output_folder, 'stops.txt'),
                             dtype={'stop_id': 'str', 'parent_station': 'str'})
    trips_data = pd.read_csv(os.path.join(output_folder, 'trips.txt'), dtype={'trip_id': 'str', 'route_id': 'str'})
    shapes_data = pd.read_csv(os.path.join(output_folder, 'shapes.txt'))
    routes = pd.read_csv(os.path.join(output_folder, 'routes.txt'), dtype={'route_id': 'str', 'agency_id': 'str',
                                                                           'route_short_name': 'str',
                                                                           'route_long_name': 'str'})

    if remove_unused_stations:
        stops_data, stops_times = _remove_unused_stations(stops_data, stops_times)

    stop_times = pd.merge(stops_times, stops_data, on='stop_id', how='left', validate='m:1')
    trips_data = pd.merge(trips_data, routes, on='route_id', how='left', validate='m:1')

    # Set indexes for faster querying and merging
    trips_data.set_index(['trip_id', 'route_id', 'direction_id'], inplace=True, drop=True, verify_integrity=True)
    stop_times.set_index(['trip_id', 'stop_id', 'stop_sequence'], inplace=True, drop=True, verify_integrity=True)
    shapes_data.set_index(['shape_id', 'shape_pt_sequence'], inplace=True, drop=True, verify_integrity=True)
    stops_data.set_index('stop_id', inplace=True, drop=True, verify_integrity=True)

    data = static_data(stop_times=stop_times, stops=stops_data, trips=trips_data, shapes=shapes_data, routes=routes)
    return data


def clean_cache() -> None:
    """Apply cleaning filters to all files in the cache, to fix issues with arrays downloaded with older versions."""

    for f in tqdm.tqdm(glob.glob(os.path.join(config.CACHE_DIR, '*feather'))):
        df = pd.read_feather(f)
        if 'ServiceAlerts' in f:
            df = getdata.unpack_jsons(df)
        getdata.sanitise_array(df)

        if df.empty:  # Feather does not support a DF without columns, so add a dummy one
            df['_'] = np.zeros(len(df), dtype=np.bool_)
        df.reset_index(drop=True).to_feather(f, compression='zstd', compression_level=9)
