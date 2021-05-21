"""Geographic-related functions."""

import math

import numpy as np
from cartopy import crs as ccrs
from geopy import distance

SWEREF99 = ccrs.TransverseMercator(central_longitude=15, scale_factor=0.9996, approx=False)
PLATE_CARREE = ccrs.PlateCarree()


def flat_distance(a: (float, float), b: (float, float)) -> float:
    """Compute the distance in metres between two points using the fast planar approximation.
    Accurate at small distances."""
    lat_a, lon_a = a
    lat_b, lon_b = b
    x = math.radians(lon_b - lon_a) * math.cos(math.radians((lat_a + lat_b) / 2))
    y = math.radians(lat_b - lat_a)
    return distance.EARTH_RADIUS * 1e3 * math.hypot(x, y)


def geodesic_distance(a: (float, float), b: (float, float)) -> float:
    """Compute the distance between two points on the globe according to WGS84"""
    return distance.distance(a, b).m

def project_points(points: np.array) -> np.array:
    """Project latitude and longitude using SWEREF99.

    Returns positions in metres."""
    transformed = SWEREF99.transform_points(PLATE_CARREE, points[:, 0], points[:, 1])[:, :2]
    return transformed
