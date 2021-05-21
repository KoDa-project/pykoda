Gotchas in the GTFS data
########################

Here is a list of some of the unexpected behaviour of the GTFS data as we encountered it.
For some of them, we provide some helping functions in the ``pykoda`` library.


``TripUpdates``
---------------
The delays given are actually estimates, and each station may get different values,
depending on when they are issued. For example, we may have an expected delay on
several stations when the bus starts the trip, and when it is getting close.


``VehiclePositions``
---------------------
Some companies include the GPS positions when the vehicles are not on a route,
parked in a depot, or travelling between trips.


``GTFSStatic``
--------------

Some stations are subdivided into many children stations, with different ``stop_id``.
For example, different platforms for the subway or the bus stop are given a different id, but have a common parent.

The Static feed lists many more stations that are actually in use at any given date.
