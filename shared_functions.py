import pandas as pd
import numpy as np
import xarray as xr
from jug import TaskGenerator
import tobac

rams_dims_lite = {
    "phony_dim_0": "p",
    "phony_dim_3": "ztn",
    "phony_dim_1": "Y",
    "phony_dim_2": "X",
}

rams_dims_anal = {
    "phony_dim_2": "ztn",
    "phony_dim_0": "Y",
    "phony_dim_1": "X",
    "phony_dim_3": "p",  # patch
    "phony_dim_4": "s",  # surf water
    "phony_dim_5": "g",  # soil levels
}


def get_rams_output(
    p, variables, dims=rams_dims_lite, latlon=True, coords=True
):
    # read in a RAMS output file, returns an xarray dataset

    if latlon or coords:
        variables = variables + ["GLAT", "GLON"]

    ds = xr.open_dataset(p, phony_dims="access", engine="h5netcdf")[variables]
    ds = ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))

    if coords:
        ds = ds.assign_coords(
            {
                "X": ds.X,
                "Y": ds.Y,
                "ztn": ds.ztn,
                "lat": (["Y", "X"], np.array(ds.GLAT)),
                "lon": (["Y", "X"], np.array(ds.GLON)),
            }
        )

    return ds


def get_xy_spacing(grid):
    # returns the grid spacing in m

    if grid == "g1":
        return 1600
    elif grid == "g2":
        return 400
    elif grid == "g3":
        return 100


def subset_data(ds, latbounds):
    # takes an xarray dataset ds from RAMS output and selects only
    # the values within the defined lat/lon boundaries
    ds = ds.where(
        (ds.GLAT >= latbounds[0])
        & (ds.GLAT <= latbounds[1])
        & (ds.GLON >= latbounds[2])
        & (ds.GLON <= latbounds[3]),
        drop=True,
    )

    return ds


@TaskGenerator
def combine_tobac_list(features_list):
    # takes a list of tobac output dataframes and combines them into one dataframe
    return tobac.utils.combine_feature_dataframes(features_list)


@TaskGenerator
def save_df(df, savePath):
    # save dataframe as pickle in savePath
    df.to_pickle(savePath)
