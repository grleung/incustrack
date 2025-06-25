import pandas as pd
import numpy as np
import xarray as xr

# from jug import TaskGenerator
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


nz = 232


def read_header(dataPath, p, nz, var="__ztn01", varname="z"):
    # fxn to read thermodynamic z from header file
    header_file_name = (
        f"{dataPath}/{p.split('/')[-1].split('.')[0][:-2]}head.txt"
    )
    with open(header_file_name) as f:
        mylist = f.read().splitlines()
    ix = mylist.index(var)
    numlines = int(mylist[ix + 1])
    coord = mylist[ix + 2 : ix + 2 + numlines]
    coord = np.array([float(x) for x in coord])
    return coord


alt = read_header(
    f"/monsoon/MODEL/LES_MODEL_DATA/V1/DRC1.1-R-V1/G3/out_30s/",
    f"a-L-2016-12-30-110000-g1.h5",
    nz=232,
)


dz = 1 / read_header(
    f"/monsoon/MODEL/LES_MODEL_DATA/V1/DRC1.1-R-V1/G3/out_30s/",
    f"a-L-2016-12-30-110000-g1.h5",
    nz=232,
    var="__dztn01",
)

rams_dims_lite = {
    "phony_dim_0": "p",
    "phony_dim_3": "Z",
    "phony_dim_1": "Y",
    "phony_dim_2": "X",
}

rams_dims_anal = {
    "phony_dim_2": "Z",
    "phony_dim_0": "Y",
    "phony_dim_1": "X",
    "phony_dim_3": "p",  # patch
    "phony_dim_4": "s",  # surf water
    "phony_dim_5": "g",  # soil levels
}


def get_rams_output(
    path: str,
    variables: [str],
    bounds: bool = None,
    dims: dict[str] = rams_dims_lite,
    subset: bool = True,
    subsetxy: bool = True,
    coords: bool = True,
) -> xr.Dataset:
    """
    Read in RAMS output data and create xarray dataset

    Arguments:
        path -- full path to file
        variables -- list of RAMS variable names (see RAMS documentation)

    Keyword Arguments:
        latbounds -- list of lat/lon bounding box coordinates (default: {None})
        dims -- names of dimensions (default: {rams_dims_lite})
        subset -- should the data be subset for given  bounds? (default: {True})
        subsetxy -- are bounds given as a list of xy points or latlon points? (default: {True})
        coords -- should coordinates be assigned? (default: {True})

    Returns:
        Read in RAMS data and optionally subset
    """
    grid = path.split("/")[-1][-5:-3]

    if ((subset) and (not subsetxy)):
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
    ds = ds.unify_chunks()

    if ((grid != "g3") & subset):
        if subsetxy:    
            ds = subset_data_xy(ds, bounds)
        else:
            ds = subset_data_latlon(ds, bounds)

    if coords:
        ds = assign_coords(ds)

    if "Z" in ds.dims:
        ds = ds.assign_coords(ztn=("Z", alt))

    if len(variables) == 1:
        ds = ds[variables[0]]

    return ds

def rename_dims(ds, dims=rams_dims_lite):
    return ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))

def assign_coords(ds):
    if "z" in ds.dims:
        c = {
            "X": ds.X,
            "Y": ds.Y,
            "Z": ds.Z
            }
    else:
        c = {
            "X": ds.X,
            "Y": ds.Y,}
        
    if 'GLAT' in list(ds.keys()):
        c['lat'] =  (["Y", "X"], np.array(ds.GLAT))
        c["lon"] =  (["Y", "X"], np.array(ds.GLON))

    return ds.assign_coords(c)


def get_xy_spacing(grid):
    # returns the grid spacing in m
    if grid == "g1":
        return 1600
    elif grid == "g2":
        return 400
    elif grid == "g3":
        return 100


def subset_data_latlon(ds, latbounds):
    # takes an xarray dataset ds from RAMS output and selects only
    # the values within the defined lat/lon boundaries
    masky = (
        ds.Y.where(
            (ds.GLAT >= latbounds.iloc[0])
            & (ds.GLAT <= latbounds.iloc[1])
            & (ds.GLON >= latbounds.iloc[2])
            & (ds.GLON <= latbounds.iloc[3])
        )
        .dropna(dim="X", how="all")
        .dropna(dim="Y", how="all")
        .values
    )
    maskx = (
        ds.X.where(
            (ds.GLAT >= latbounds.iloc[0])
            & (ds.GLAT <= latbounds.iloc[1])
            & (ds.GLON >= latbounds.iloc[2])
            & (ds.GLON <= latbounds.iloc[3])
        )
        .dropna(dim="X", how="all")
        .dropna(dim="Y", how="all")
        .values
    )

    ds = ds.sel(
        Y=slice(
            np.nanmin(masky).astype("int"), np.nanmax(masky).astype("int") + 1
        ),
        X=slice(
            np.nanmin(maskx).astype("int"), np.nanmax(maskx).astype("int") + 1
        ),
    )
    return ds


def subset_data_xy(ds, xybounds):
    # takes an xarray dataset ds from RAMS output and selects only
    # the values within the defined x/y gridpoint boundaries
    ds = ds.sel(
        Y=slice(
            xybounds[0],xybounds[1]
        ),
        X=slice(
            xybounds[2],xybounds[3]
        ),
    )
    return ds

def save_files(out, savePath):
    out["time"] = pd.to_datetime(out["timestr"])
    # save dataframe as a parquet in savePath
    out.to_parquet(savePath, engine="pyarrow")


def compute_cond(ds, return_dens=False):
    ds = ds.assign(PRES=p00 * (ds.PI / cp) ** (cp / rd))
    ds = ds.assign(TEMP=ds.THETA * (ds.PI / cp))
    ds = ds.assign(DENS=ds.PRES / (rd * ds.TEMP * (1 + (0.61 * ds.RV))))

    ds = ds.assign(COND=(ds.RCP + ds.RSP + ds.RPP) * ds.DENS)

    if return_dens:
        ds = ds[["COND", "DENS"]]
    else:
        ds = ds["COND"]
    return ds

def compute_dens(ds):
    ds = ds.assign(PRES=p00 * (ds.PI / cp) ** (cp / rd))
    ds = ds.assign(TEMP=ds.THETA * (ds.PI / cp))
    ds = ds.assign(DENS=ds.PRES / (rd * ds.TEMP * (1 + (0.61 * ds.RV))))
    ds = ds["DENS"]
    return ds


def compute_pcp(ds):
    ds = ds.assign(
        PCPT=(
            ds.PCPRR
            + ds.PCPRP
            + ds.PCPRS
            + ds.PCPRA
            + ds.PCPRG
            + ds.PCPRH
            + ds.PCPRD
        )
        * 3600
    )

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
    "WP_BUOY_THETA",
]
