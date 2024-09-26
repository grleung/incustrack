import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from scipy.ndimage import labeled_comprehension, maximum_position, sum_labels, maximum, mean, minimum
import dask
import dask.distributed as dd
from dask.diagnostics import ResourceProfiler

client = dd.Client("updraft:8786")
client.upload_file("shared_functions.py")

from shared_functions import (
    get_rams_output,
    read_header,
    get_xy_spacing,
    subset_data,
    combine_tobac_list,
    rams_dims_lite,
    all_var,
    save_files,
    compute_cond,
    compute_pcp,
    p00,
    rd,
    cp,
)

def get_footprint(arr):
    return(arr.max(axis=(0)).sum().compute())

outbounds = pd.read_pickle(f"/tempest/gleung/incustrack/bounds.pkl")

def get_masked_statistics(sub, grid, dxy, latbounds, dataPath, tobacPath):
    fts = sub.feature.unique()
    time = pd.to_datetime(sub.timestr.iloc[0])

    ds = get_rams_output(
        f"{dataPath}/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}-{grid}.h5",
        variables=[
            "PCPRR",
            "PCPRP",
            "PCPRS",
            "PCPRA",
            "PCPRG",
            "PCPRH",
            "PCPRD",
            "WP",
            "RCP",
            "RSP",
            "RPP",
            "PI",
            "THETA",
            "RV",
        ],
        latbounds=latbounds,
        latlon=True,
        coords=True,
    )

    temp = compute_cond(ds, return_dens=True)

    ds = ds.assign(COND=temp['COND'], DENS=temp['DENS'],PCPT=compute_pcp(ds))

    ds = ds[["COND", "PCPT", "WP", "DENS"]]

    cond_mask = xr.open_dataset(
        f"{tobacPath}/cond_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
        engine="h5netcdf",
        chunks="auto",
    )

    w_mask = xr.open_dataset(
        f"{tobacPath}/w_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
        engine="h5netcdf",
        chunks="auto",
    )

    pcp_mask = xr.open_dataset(
        f"{tobacPath}/pcp_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
        engine="h5netcdf",
        chunks="auto",
    )

    cond_mask = cond_mask.assign(alt=(("ztn"), alt / 1000))
    cond_mask = cond_mask.assign(dz=(("ztn"), dz))

    shape = (cond_mask.segmentation_mask / cond_mask.segmentation_mask).fillna(1)

    sub["CTH"] = labeled_comprehension(
        cond_mask.alt
        * shape,
        cond_mask.segmentation_mask,
        fts,
        np.nanmax,
        np.float64,
        np.nan,
    )

    sub["CBH"] = labeled_comprehension(
        cond_mask.alt
        * shape,
        cond_mask.segmentation_mask,
        fts,
        np.nanmin,
        np.float64,
        np.nan,
    )

    sub["condensate_volume"] = sum_labels(
        cond_mask.dz* shape,
        cond_mask.segmentation_mask,
        fts,
    )* dxy* dxy/ (1000 * 1000 * 1000)

    sub["condensate_count"] = sum_labels(
        shape,
        cond_mask.segmentation_mask,
        fts,
    )

    sub["updraft_topalt"] = maximum(
        cond_mask.alt * shape,
        w_mask.segmentation_mask,
        fts,
    )

    sub["updraft_botalt"] = minimum(
        cond_mask.alt * shape,
        w_mask.segmentation_mask,
        fts,
    )

    sub["updraft_volume"] = sum_labels(
        cond_mask.dz* shape,
        w_mask.segmentation_mask,
        fts,
    ) * dxy* dxy/ (1000 * 1000 * 1000)

    sub["updraft_count"] = sum_labels(
        shape,
        w_mask.segmentation_mask,
        fts,)

    sub["precip_area"] = sum_labels(
        (pcp_mask.segmentation_mask / pcp_mask.segmentation_mask),
        pcp_mask.segmentation_mask,
        fts,
    )*dxy * dxy/ (1000 * 1000)

    cmf = cond_mask.dz * dxy * dxy * ds.DENS * ((ds.WP).where(ds.WP > 0))

    sub["cmf_total"] = labeled_comprehension(
        cmf,
        cond_mask.segmentation_mask,
        fts,
        np.nansum,
        np.float64,
        np.nan
    )

    sub["cmf_mean"] = labeled_comprehension(
        cmf,
        cond_mask.segmentation_mask,
        fts,
        np.nanmean,
        np.float64,
        np.nan,
    )

    sub["cmf_max"] = labeled_comprehension(
        cmf,
        cond_mask.segmentation_mask,
        fts,
        np.nanmax,
        np.float64,
        np.nan,
    )


    sub["cmfmax_loc"] = maximum_position(
        cmf.fillna(0),
        cond_mask.segmentation_mask,
        fts
    )

    sub["w_mean"] = mean(
        ds.WP,
        w_mask.segmentation_mask,
        fts,
    )

    sub["w_max"] = maximum(
        ds.WP,
        w_mask.segmentation_mask,
        fts,
    )

    sub["wmax_loc"] = maximum_position(
        ds.WP,
        w_mask.segmentation_mask,
        fts
    )
    

    sub["pcp_mean"] = mean(
        ds.PCPT,
        pcp_mask.segmentation_mask,
        fts,
    )

    sub["pcp_max"] = maximum(
        ds.PCPT,
        pcp_mask.segmentation_mask,
        fts,
    )

    sub['w_footprint'] =  [dxy * dxy *get_footprint((w_mask.segmentation_mask==ft).data)/1000**2 for ft in fts]

    sub['cond_footprint'] = [dxy * dxy * get_footprint((cond_mask.segmentation_mask==ft).data)/1000**2 for ft in fts]

    sub['cmfmax_alt'] = alt[pd.DataFrame(sub.cmfmax_loc.to_list())[0]]/1000
    sub.loc[sub[sub.cmf_max.isnull()].index,'cmfmax_alt'] = np.nan
    sub['wmax_alt'] = alt[pd.DataFrame(sub.wmax_loc.to_list())[0]]/1000
    

    return sub


runs = ['DRC1.1-R-V1',"PHI1.1-R-V1", "PHI2.1-R-V1",'AUS1.1-R-V1',"ARG1.2-R-V1", ]
grids = ['g2']

for grid in grids:
    for run in runs:
        latbounds = outbounds.loc[run].values
    
        print(run, grid)

        dxy=get_xy_spacing(grid)

        dataPath = f"/monsoon/MODEL/LES_MODEL_DATA/V1/{run}/G3/out_30s/"
        tobacPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}"

        out = pd.read_parquet(f"{tobacPath}/combined_segmented_tracks.pq")

        alt = read_header(
            dataPath,
            f"a-L-{out['time'].iloc[0].strftime('%Y-%m-%d-%H%M%S')}-g3.h5",
            nz=232,
        )

        dz = 1 / read_header(
            dataPath,
            f"a-L-{out['time'].iloc[0].strftime('%Y-%m-%d-%H%M%S')}-g3.h5",
            nz=232,
            var="__dztn01",
        )

        x = client.map(
            get_masked_statistics,
            [out[out.frame==frame] for frame in sorted(out.frame.unique())],
            grid=grid,
            dxy=get_xy_spacing(grid),
            latbounds=latbounds,
            dataPath=dataPath,
            tobacPath=tobacPath,
        )
        x = client.gather(x)

        out = pd.concat(x)

        out.to_parquet(f"{tobacPath}/combined_segmented_tracks_statistics.pq")
