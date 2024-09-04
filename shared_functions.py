import pandas as pd
import numpy as np
import xarray as xr
from jug import TaskGenerator
import tobac
import dask
import os

# physical constants
g = 9.8065
eps = 0.622
cp = 1004
rd = 287
p00 = 100000
rgas = 287
lv = 2.5e6

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
    path, variables, latbounds=None, dims=rams_dims_lite, latlon=True, coords=True
):
    time = pd.to_datetime(path.split("/")[-1][4:-6])
    grid = path.split("/")[-1][-5:-3]

    if latlon or coords:
        drop_var = [v for v in all_var if v not in variables + ["GLAT", "GLON"]]
    else:
        drop_var = [v for v in all_var if v not in variables]

    ds = xr.open_dataset(
        path,
        phony_dims="access",
        engine="h5netcdf",
        chunks="auto",
        drop_variables=drop_var,
    )

    ds = rename_dims(ds, dims)

    if grid != "g3":
        ds = subset_data(ds, latbounds)

    if coords:
        ds = assign_coords(ds)

    if len(variables) == 1:
        ds = ds[variables[0]]

    return ds


def rename_dims(ds, dims=rams_dims_lite):
    return ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))


def assign_coords(ds):
    if "ztn" in ds.dims:
        c = {
            "X": ds.X,
            "Y": ds.Y,
            "ztn": ds.ztn,
            "lat": (["Y", "X"], np.array(ds.GLAT)),
            "lon": (["Y", "X"], np.array(ds.GLON)),
        }
    else:
        c = {
            "X": ds.X,
            "Y": ds.Y,
            "lat": (["Y", "X"], np.array(ds.GLAT)),
            "lon": (["Y", "X"], np.array(ds.GLON)),
        }
    return ds.assign_coords(c)


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
    ds = (
        ds.where(
            (ds.GLAT >= latbounds[0])
            & (ds.GLAT <= latbounds[1])
            & (ds.GLON >= latbounds[2])
            & (ds.GLON <= latbounds[3]),
        )
        .dropna(dim="X", how="all")
        .dropna(dim="Y", how="all")
    )

    return ds


def combine_tobac_list(features_list):
    # takes a list of tobac output dataframes and combines them into one dataframe
    return tobac.utils.combine_feature_dataframes(features_list)


def save_files(out, savePath):
    out["time"] = pd.to_datetime(out["timestr"])
    # save dataframe as a parquet in savePath
    out.to_parquet(savePath, engine="pyarrow")


def compute_cond(ds):
    ds = ds.assign(PRES=p00 * (ds.PI / cp) ** (cp / rd))
    ds = ds.assign(TEMP=ds.THETA * (ds.PI / cp))
    ds = ds.assign(DENS=ds.PRES / (rd * ds.TEMP * (1 + (0.61 * ds.RV))))

    ds = ds.assign(COND=(ds.RCP + ds.RSP + ds.RPP) * ds.DENS)

    ds = ds["COND"]
    return ds

def compute_pcp(ds):
    ds = ds.assign(PCPT=(
            ds.PCPRR
            + ds.PCPRP
            + ds.PCPRS
            + ds.PCPRA
            + ds.PCPRG
            + ds.PCPRH
            + ds.PCPRD
        )
        * 3600)

    ds = ds["PCPT"]
    return ds


all_var = [
    "CAN_TEMP",
    "CAP",
    "CCP",
    "CDP",
    "CGP",
    "CHP",
    "CIFNP",
    "CN1NP",
    "CPP",
    "CRP",
    "CSP",
    "FTHRD",
    "GLAT",
    "GLON",
    "LATHEATFRZ",
    "LATHEATVAP",
    "PATCH_ROUGH",
    "PCPRA",
    "PCPRD",
    "PCPRG",
    "PCPRH",
    "PCPRP",
    "PCPRR",
    "PCPRS",
    "PI",
    "PP",
    "Q6",
    "Q7",
    "RAP",
    "RCP",
    "RDP",
    "REGEN_AERO1_NP",
    "RGP",
    "RHP",
    "RLONTOP",
    "RPP",
    "RRP",
    "RSP",
    "RTP",
    "RV",
    "SFLUX_R",
    "SFLUX_T",
    "SOIL_ENERGY",
    "SOIL_WATER",
    "THETA",
    "TOPT",
    "TSTAR",
    "UP",
    "USTAR",
    "VEG_TEMP",
    "VP",
    "WP",
    "WP_ADVDIF",
    "WP_BUOY_COND",
]
