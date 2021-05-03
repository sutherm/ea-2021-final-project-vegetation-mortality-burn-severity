"""Defines utility functions for working with geospatial data"""
import hashlib
import json
import os
import re
import tempfile
import zipfile

import geopandas as gpd
import numpy as np
import requests
import xarray as xr




def checksum(path, size=8192):
    """Generates MD5 checksum for a file

    Parameters
    ----------
    path: str
        path of file to hash
    size: int
        size of block. Must be multiple of 128.

    Returns
    -------
        MD5 checksum of file
    """
    if size % 128:
        raise ValueError('Size must be a multiple of 128')
    with open(path, "rb") as f:
        md5_hash = hashlib.md5()
        while True:
            chunk = f.read(size)
            if not chunk:
                break
            md5_hash.update(chunk)
        return md5_hash.hexdigest()


def checksum_walk(path):
    """Walks path and generates a MD5 checksum for each file

    Parameters
    ----------
    path: str 
        path of file to hash
        
    Returns
    -------
    None
    """
    for root, dirs, files in os.walk(path):
        if files:

            # Look for existing checksums
            checksum_path = os.path.join(root, "checksums.json")
            try:
                with open(checksum_path, "r") as f:
                    checksums = json.load(f)
            except IOError:
                checksums = {}

            # Checksum each data file
            for fn in files:
                if (
                    fn != "checksums.json"
                    and os.path.splitext(fn)[-1] != ".md5"
                ):
                    try:
                        checksums[fn]
                    except KeyError:
                        checksums[fn] = checksum(os.path.join(root, fn))

            # Save checksums as JSON
            with open(checksum_path, "w") as f:
                json.dump(checksums, f, indent=2)
    
    
def download_file(url, path):
    """Downloads file at url to path
    
    Parameters
    ----------
    url: str
        url to download
    path: str
        path to download to
        
    Returns
    -------
    None
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)


def join_and_makedirs(*args):
    """Joins path and creates parent directories if needed
    
    
    Parameters
    ----------
    args: str
        one or more path segments
    
    Returns
    -------
    str
        the joined path
    """
    path = os.path.join(*args)
    
    # Get parent directory if file
    parents = path
    if re.search(r"\.[a-z]{2,4}$", parents, flags=re.I):
        parents = os.path.dirname(path)
        
    # Create parent directories as needed
    try:
        os.makedirs(parents)
    except OSError:
        pass

    return path


def zip_shapefile(gdf, path):
    """Exports GeoDataFrame to shapefile and zips it
    
    The resulting zip file can be use to search USGS Earth Explorer
    
    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        the data frame to export
    path: str
        the path to the zip file
        
    Returns
    -------
    None
    """
    
    # Write shapefile files to a temporary directory
    tmpdir = tempfile.TemporaryDirectory()
    stem = os.path.basename(os.path.splitext(path)[0])
    gdf.to_file(os.path.join(tmpdir.name, f"{stem}.shp"))
    
    # Zip the temporary files
    with zipfile.ZipFile(path, "w") as f:
        for fn in os.listdir(tmpdir.name):   
            f.write(os.path.join(tmpdir.name, fn), fn, zipfile.ZIP_DEFLATED)


def create_sampling_mask(xda, counts, boundary, seed=None, path=None):
    """Creates mask for a sample of a data array"""

    # Calculate sample size based on area of shape relative to envelope
    pct_area = boundary.area / boundary.geometry.envelope.area
    scalar = 1.1 / pct_area
    pool_size = int(scalar * sum(counts.values()))

    # Clip data array to AOI and create a mask
    xda = xda.rio.clip(boundary.geometry)
    boundary_mask = np.where(np.isfinite(xda.values), True, False)

    # Create array of x and y for given array
    xv, yv = np.meshgrid(np.arange(xda.sizes["x"]), np.arange(xda.sizes["y"]))
    xy = np.column_stack((xv.ravel(), yv.ravel()))
    
    # Create a sample pool large enough to accommodate masks in counts
    rng = np.random.default_rng(seed)
    pool = rng.choice(xy, pool_size, replace=False).tolist()

    # Add each mask as a layer in an xarray
    masks = {}
    for name, count in counts.items():
        
        # Pull samples from the pool to use for this mask
        sample_size = int(scalar * count)
        sample, pool = pool[:sample_size], pool[sample_size:]
        
        # Build mask based on the sample
        mask = np.full(xda.shape[-2:], 0)
        for (col, row) in rng.choice(xy, sample_size, replace=False):
            mask[row][col] = 1

        # Pool sizes are padded by 10%, so the sample is larger than needed.
        # Count pixels that fall into the boundary, then remove points to
        # get down to the exact number desired.
        mask = np.where(boundary_mask, mask, 0)
        rows, cols = np.where(mask == 1)
        xy = list(zip(cols, rows))
        for (col, row) in rng.choice(xy, len(xy) - count, replace=False):
            mask[row][col] = 0
            
        masks[name] = mask
    
    # Create data array containing the masks
    arrs = []
    for name, mask in masks.items():
        arrs.append(xr.DataArray(mask,
                                 coords={"y": xda.y, "x": xda.x},
                                 dims=["y", "x"]))
        arrs[-1]["band"] = len(arrs)

    sampling_mask = xr.concat(arrs, dim="band")
    sampling_mask.attrs["long_name"] = list(counts.keys())
    sampling_mask["spatial_ref"] = 0
    sampling_mask["spatial_ref"].attrs = xda["spatial_ref"].attrs
    
    # Write mask to raster file if path given
    if path:
        sampling_mask.rio.to_raster(path)
       
    # Convert to a true-false mask
    return xr.where(sampling_mask == 1, True, False)


def load_nifc_fires(*fire_ids, crs=None, **kwargs):
    """Loads fires matching the given IDs from NIFC shapefile"""
    
    # Load the NIFC fire shapefile
    nifc_fires_path = os.path.join(
        "custom",
        "nifc_fire_perimeters",
        "US_HIST_FIRE_PERIMTRS_2000_2018_DD83.shp"
    )
    nifc_fires = gpd.read_file(nifc_fires_path, **kwargs)
    
    # Reproject to given CRS if needed
    if crs and nifc_fires.crs != crs:
        nifc_fires.to_crs(crs, inplace=True)  

    # Limit dataframe to required columns
    nifc_fires = nifc_fires[[
        "agency",
        "uniquefire",
        "incidentna",
        "complexnam",
        "fireyear",
        "perimeterd",
        "gisacres",
        "geometry"
    ]]
    
    if fire_ids:
        return nifc_fires[nifc_fires.uniquefire.isin(fire_ids)]
    return nifc_fires