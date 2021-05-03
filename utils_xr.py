"""Defines decorators and plotting functions for xarrays"""

from functools import wraps
import os
import re

import numpy as np
import earthpy.plot as ep
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import plotting_extent as rasterio_plotting_extent
import rioxarray as rxr
import xarray as xr




def as_xarray(func):
    """Wraps a non-xarray function so that metadata is maintained"""

    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)

        # Copy metadata from original object and add band if needed
        xobj = copy_xr_metadata(args[0], result)
        if "band" not in xobj.coords and "band" not in xobj.dims:
            xobj = add_dim(xobj, dim="band", coords={"name" : ["result"]})

        return xobj
    return wrapped


def plot_xarray(func):
    """Wraps an xarray object to allow plotting with earthpy"""

    @wraps(func)
    def wrapped(*args, **kwargs):

        # Convert first argument to a masked array to plot with earthpy
        args = list(args)
        arr = to_numpy_array(args[0])

        # Automatically assign extent for plots if rio accessor is active
        if func.__name__.startswith("plot_"):
            try:
                kwargs.setdefault("extent", plotting_extent(args[0]))
            except AttributeError:
                # Fails if rio accessor has not been loaded
                raise

        # HACK: Masked arrays cannot be stretched because they are not
        # handled intuitively by the np.percentile function used by the
        # earthpy internals. To get around that, the decorator forces NaN
        # values to 0 when stretch is True.
        if kwargs.get("stretch"):
            pct_clip = np.nanmedian(arr)
            arr = to_numpy_array(args[0].fillna(0))
        else:
            arr = np.ma.masked_invalid(arr)

        return func(arr, *args[1:], **kwargs)
    return wrapped


@plot_xarray
def hist(*args, **kwargs):
    """Plots histogram based on an xarray object"""
    return ep.hist(*args, **kwargs)


@plot_xarray
def plot_bands(*args, **kwargs):
    """Plots bands based on an xarray object"""
    return ep.plot_bands(*args, **kwargs)


@plot_xarray
def plot_rgb(*args, **kwargs):
    """Plots RGB based on an xarray object"""
    return ep.plot_rgb(*args, **kwargs)


def add_dim(xobj, dim="band", coords=None):
    """Adds an index dimension to an array

    Parameters
    ---------
    xobj: xarray.DataArray or xarray.Dataset
        an array without an index dimension
    dim: str
        the name of the index dimension
    coords: dict of list-like
        list of names for the bands in the given xarray. The length of each
        list must match that of the array.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
       Array with band as a dimensional coordinate or dataset with band as keys
    """

    # Convert dataset to array
    is_dataset = False
    if isinstance(xobj, xr.Dataset):
        xobj = xobj.to_array(dim=dim)

    # Check shape to see if it contains only one band
    if len(xobj.shape) == 2:
        xobj = [xobj]

    # Assign band
    layers = []
    for arr in xobj:
        arr[dim] = len(layers)
        layers.append(arr)
    new_xobj = xr.concat(layers, dim=dim)

    # Map any provided names
    if coords:
        coords = {k: (dim, list(v)) for k, v in coords.items()}
        new_xobj = new_xobj.assign_coords(**coords)

    return new_xobj.to_dataset(dim=dim) if is_dataset else new_xobj


def copy_xr_metadata(xobj, other):
    """Copies metadata from one xarray object to another

    Parameters
    ---------
    xarr: xarray.DataArray or xarray.Dataset
        the array/dataset to copy metadata from
    other: numpy.array or similar
        the object to copy metadata to

    Returns
    -------
    xarray.DataArray or xarray.Datset
       Copy of other converted to type of xobj with metadata applied
    """
    if isinstance(xobj, xr.DataArray):
        return copy_array_metadata(xobj, other)
    return copy_dataset_metadata(xobj, other)


def copy_array_metadata(xarr, other):
    """Copies metadata from an array to another object

    Looks at the shape and length of xarr and other to decide which
    metadata to copy.

    Parameters
    ---------
    xarr: xarray.DataArray
        the array to copy metadata from
    other: numpy.array or similar
        the object to copy metadata to

    Returns
    -------
    xarray.DataArray
       Copy of other with metadata applied
    """

    # Convert a list, etc. to an array
    if isinstance(other, (list, tuple)):
        other = np.array(other)

    # If arrays have the same shape, copy all metadata
    if xarr.shape == other.shape:
        return xarr.__class__(other, dims=xarr.dims, coords=xarr.coords)

    # If arrays have the same number of layers, copy scalar coordinates
    # and any other coordinates with same the length as the array
    if len(xarr) == len(other):
        coords = {
            k: v for k, v in xarr.coords.items()
            if not v.shape or v.shape[0] == len(xarr)
        }
        dims = [d for d in xarr.dims if d in coords]
        return xarr.__class__(other, dims=dims, coords=coords)

    # If arrays have the same dimensions, copy spatial and scalar coordinates
    xarr_sq = xarr.squeeze()
    other_sq = other.squeeze()
    if xarr_sq.shape == other_sq.shape:
        coords = {k: v for k, v in xarr_sq.coords.items()
                  if k not in xarr_sq.dims}
        return xarr.__class__(
            other_sq, dims=xarr_sq.dims, coords=xarr_sq.coords)

    raise ValueError("Could not copy xr metadata")


def copy_dataset_metadata(xdat, other):
    """Copies metadata from a dataset to another object

    Parameters
    ---------
    xarr: xarray.Dataset
        the dataset to copy metadata from
    other: numpy.array or similar
        the object to copy metadata to

    Returns
    -------
    xarray.Dataset
       Copy of other with metadata applied
    """
    xarr = xdat.to_array(dim="band")
    return copy_array_metadata(xarr, other).to_dataset(dim="band")


def iterarrays(xobj):
    """Iterates through an xarray object

    Parameters
    ---------
    xarr: xarray.DataArray or xarray.Dataset
        the object to iterate

    Returns
    -------
    iterable
        list or similar of the children of the given object
    """
    if isinstance(xobj, xr.DataArray):
        return xobj if len(xobj.shape) > 2 else [xobj]
    return xobj.values()


def to_numpy_array(obj, dim="band"):
    """Converts an object to a numpy array

    Parameters
    ----------
    obj: array-like
        a numpy array, xarray object or any other object that can converted
        to a numpy array using numpy.array()
    dim: str
        the name of dimension of the new array when converting a dataset

    Returns
    -------
    numpy.array
        an array based on the given object
    """
    if isinstance(obj, xr.Dataset):
        xobj = xobj.to_array(dim=dim)
    if isinstance(obj, xr.DataArray):
        return obj.values
    if isinstance(obj, (list, tuple)):
        return np.array(obj)
    return obj


def plotting_extent(xobj):
    """Calculates plotting extent for an xarray object for matplotlib

    Parameters
    ----------
    xobj: xarray.DataArray or xarray.Dataset
        the xarray object to scale

    Returns
    -------
    tuple of float
        left, right, bottom, top
    """
    for xarr in iterarrays(xobj):
        return rasterio_plotting_extent(xarr, xarr.rio.transform())


def open_raster(path, crs=None, crop_bound=None, nodata=None):
    xda = rxr.open_rasterio(path, masked=True)
    
    # Write nodata value
    if xda.rio.nodata is None and nodata is not None:
        xda = xda.rio.write_nodata(nodata)
    
    # Reproject to CRS
    if crs is not None and xda.rio.crs != crs:
        xda = xda.rio.reproject(crs)
    
    # Reproject to CRS
    if crop_bound is not None:
        if crop_bound.crs != xda.rio.crs:
            crop_bound = crop_bound.to_crs(xda.rio.crs)
        xda = xda.rio.clip(crop_bound, drop=True, from_disk=True)
    
    return xda

    
def reproject_match(xda, match_xda, **kwargs):
    """Forces reprojection to use exact x, y coordinates of original array
    
    Reprojection can produce small differences in coordinates (like -4e10) that
    break an exact xarray.align.
    """
    
    # Align data type
    if xda.dtype != match_xda.dtype:
        xda = xda.astype(match_xda.dtype)
    
    reproj = xda.rio.reproject_match(match_xda, **kwargs)
    reproj = xr.DataArray(reproj.values,
                          coords={"band": xda.band,
                                  "y": match_xda.y,
                                  "x": match_xda.x,
                                  "spatial_ref": match_xda.spatial_ref},
                          dims=["band", "y", "x"])

    #reproj.attrs = xda.attrs 

    return reproj


def find_scenes(src):
    """Finds and groups all files that are part of a scene"""
    patterns = {
        "landsat": r"((?<=band)(\d)|[a-z]+_qa)\.tif$",
        "sentinel": r"_B([01][0-9])\.jp2$",
    }
    
    scenes = {}
    for root, dirs, files in os.walk(src):
        for key, pattern in patterns.items():
            for fn in files:
                         
                try:
                    band = re.search(pattern, fn).group(1).lstrip("0")
                except AttributeError:
                    pass
                else:
                    # Get scene using the path to the file
                    segments = root.split(os.sep)
                    scene = segments.pop(-1)
                    while not re.search(r"\d", scene):
                        scene = segments.pop(-1)

                    scenes.setdefault(scene, {})[band] = os.path.join(root, fn)
    
    return scenes

        
def stack_scene(scene, align_to=None):
    """Stacks all files that are part of a scene"""
    layers = []
    attrs = {}
    for band in sorted(scene, key=lambda s: s.zfill(16)):
        layer = rxr.open_rasterio(scene[band], masked=True)
        
        # Align scene if align_to is given
        if align_to is not None:
            layer = reproject_match(layer, align_to)
        
        layers.append(layer)

        # Update band number
        del layers[-1]["band"]
        layers[-1] = layers[-1].squeeze()
        layers[-1]["band"] = len(layers)

        attrs.setdefault("band_name", []).append(band)
        attrs.setdefault("long_name", []).append(os.path.basename(scene[band]))

    xda = xr.concat(layers, dim="band")
    for key, val in attrs.items():
        xda.attrs[key] = val
    
    return xda