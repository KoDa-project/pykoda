"""Helper functions related to plotting.
"""
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import OSM
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature

from typing import Optional


def setup_mpl() -> None:
    """Set up rcParams for Matplotlib."""

    plt.rcParams['figure.constrained_layout.use'] = True


def add_basemap(zoom_level: int = 14, ax: Optional[GeoAxes] = None) -> None:
    """Add a background map image to the plot.

    If ax is not passed, it will use the current axis.
    """
    if ax is None:
        ax = plt.gca()

    openstreetmap_tiles = OSM()
    try:
        ax.add_image(openstreetmap_tiles, zoom_level)
    except AttributeError:
        raise ValueError('The axis must be declared with a projection to add a basemap')


def add_water(ax: Optional[GeoAxes] = None) -> None:
    if ax is None:
        ax = plt.gca()

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='k', facecolor='none')
    minor_islands_10m = cfeature.NaturalEarthFeature('physical', 'minor_islands', '10m',
                                                     edgecolor='k', facecolor='none')
    rivers_10m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
                                              edgecolor='b', facecolor='none')
    rivers_europe_10m = cfeature.NaturalEarthFeature('physical', 'rivers_europe', '10m',
                                              edgecolor='b', facecolor='none')
    lakes_10m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                             edgecolor='b', facecolor='b', alpha=0.1)
    ocean_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                             edgecolor='b', facecolor='b', alpha=0.1)
    ax.add_feature(land_10m)
    ax.add_feature(minor_islands_10m)
    ax.add_feature(rivers_10m)
    ax.add_feature(rivers_europe_10m)
    ax.add_feature(lakes_10m)
    # ax.add_feature(ocean_10m)
