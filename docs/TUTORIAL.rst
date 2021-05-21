Pykoda
######

The ``pykoda`` library is an interface to the KoDa API
with a collection of utilities for analysis and machine learning applications.

Its main utility is handling the downloading, parsing, and caching of historical data.


The KoDa Data
--------------

KoDa is an unified source of public transportation data, as provided by the transit authorities.
The data follows the GTFS standard, and it is stored in four different feeds:

1. ``GTFSStatic`` contains the information in the schedule: from stops locations to scheduled departure and arrivals.
2. ``VehiclePositions`` real time GPS position of the vehicles and the route they are into. They are updated every 1-2 seconds.
3. ``TripUpdates`` real time information on arrival and departure times, delays, etc.
4. ``ServiceAlerts`` notices about service interruptions, construction, etc.

Not all authorities provide all the feeds.

The companies that are providing data at the moment are:

* Dintur - Västernorrlands län.
* DT - Dalatrafik.
* KLT - Kalmar länstrafik.
* Krono - Kronobergs Länstrafik.
* Otraf - Östgötatrafiken.
* SJ, Snälltåget, and TågAB.
* Skane - Skånetrafiken.
* SL - Stockholm län.
* UL - Uppsala län.
* Varm - Värmlandstrafik and Karlstadbuss.
* VT - Västtrafik.
* XT - X-trafik (Gävleborgs län).


Dependencies
-------------

The most important dependencies are::

    ey
    pandas
    cartopy
    protobuf
    pyarrow

All will be automatically installed by ``pip``.


Configuring
-----------
Some default options can be overridden through the Pykoda config file.
Its location is OS-dependent, you can find it with::

    import appdirs
    appdirs.user_config_dir('pykoda')

This is a ``ConfigParser`` file, with currently only one section, ``[all]``,
and the following settings:

* ``api_key`` the user key to access the KoDa data. This is required to download data using the v2 API.
* ``cache_dir`` directory to save the downloaded and parsed data. Default: ``appdirs.user_cache_dir('pykoda')``
* ``n_cpu`` number of threads to use for embarrassingly parallel tasks, such as parsing data. Default: all CPUs.

.. warning::
    The v1 of the API is considered deprecated and will be turned off in the near future.
    In order to use v2, you need to get a key.
    Information on how to do it will be provided on the website.


Components
-----------

The library is split into separate modules.

``pykoda.datautils``
********************
This module is used to parse and query the data. It will automatically download what it needs.
Following the nature of the KoDa API, the data is stored per company and feed, grouped by the hour.

The most important function is::

    get_data_range(feed: str, company: str, start_date: str, start_hour: int = 0, end_date: str = None,
                       end_hour: int = 23, query: (str, None) = None, merge_static: bool = False,
                       clear_duplicates: bool = True) -> pd.DataFrame:


It will return a dataframe with the dta for the corresponding feed and company between the given dates.
Optionally, it can combine the information with the static data through the merge_static flag.

If we are interested in only a narrow subsection of the data, we can restrict memory usage by passing a query,
which will be applied to each hourly slice of data as they are loaded.

``clear_duplicates`` will keep only the latest available information for each stop.
The ``TripUpdates`` feed may issue several delay estimates for future stops, with this option only the last one will be kept.
Disable to access to the full historical information, for example, to train short-term forecasting models.

The lower-level API is given by ``pykoda.datautils.get_data`` and ``pykoda.datautils.get_static_data``.
They will download the files from the KoDa API when necessary, parse and clean the data
- unifying the keys and data types according to the standard -
and store it in the cache folder. Note that they do not actually return the data itself,
it must always be read from the cache.
Their signatures are::

    get_data(date: str, hour: (int, str), feed: str, company: str, output_file: (str, None) = None) -> None
    get_static_data(date: str, company: str, outfolder: (str, None) = None) -> None

``pykoda.geoutils``
******************************
Utilities related to geography, such as projections and geodesic calculations.

``pykoda.plotutils``
**********************
Basic plotting helper funcitons.
Currently, just implementing the
experimental constrained layout, and adding OSM basemap to existing plots.

``pykoda.graphutils``
*********************
Functions to transform the GTFS data into a graph format.


CLI
****
We also provide two command line scripts, ``koda_getstatic`` to download and cache
static data, and ``koda_getdata`` for all the other feeds.


Expanding the library
---------------------
If you can think of any improvement or functionality, feel free to request it, or even better, submit a PR with the code.


Examples
---------

The best way to understand how to use this library is to peruse the ``experiments`` folder,
where we have written examples on different tasks, from machine learning to data visualisation.
