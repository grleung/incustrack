# For questions, contact Bee (gabrielle.leung@colostate.edu)

# Import some shared libraries
import os
import dask.distributed as dd
from dask_jobqueue import SLURMCluster
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import sys
from tobac.utils import get_statistics_from_mask
from dask_memusage import install

# spin up SLURM cluster
cluster = SLURMCluster(
    cores=30,
    processes=15,
    memory="500GB",
    account="incus",
    walltime="56:00:00",
    scheduler_options={"dashboard_address": f":{sys.argv[1]}"},
    job_extra_directives=["--partition=all", "--job-name=tobac-statistics"],
)

cluster.scale(jobs=5)

# change this address depending on your scheduler address
client = dd.Client(cluster)
install(cluster.scheduler, "/home/gleung/memusage-stats-new.csv")

client.upload_file("shared_functions.py")
from shared_functions import (
    get_rams_output,
    get_xy_spacing,
    save_files,
    compute_dens,
)

# Define the paths to INCUS data and where to save output
ver = "V1"  # version of INCUS simulation dataset
modelPath = f"/monsoon/MODEL/LES_MODEL_DATA/{ver}/"
outPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/{ver}/"

runs = [
   # "ARG1.1-R-V1",
   # "ARG1.2-R-V1",
   # "AUS1.1-R-V1",
   # "BRA1.1-R-V1",
   # "BRA2.1-R-V1",
   # "PHI1.1-R-V1",
   # "USA1.1-R-V1",
   # "WPO1.1-R-V1",
   # "WPO1.1-RPR-V1",
   # "SIO1.1-R-V1",
   # "USA3.1-R-V1",
   # "PHI2.1-R-V1",
    #"DRC1.1-R-V1",
    "DRC1.1-RCR-V1",
     "SAU1.1-R-V1",

]
grids = ["g2"]

def get_statistics(path, run, grid):
    from shared_functions import dz

    dxy = get_xy_spacing(grid)

    time = pd.to_datetime(path[4:-3])

    xybounds = pd.read_parquet("/tempest/gleung/incustrack/xybounds.pq").loc[run,grid]
    ds = get_rams_output(
        f"{modelPath}/{run}/G3/out_30s/{path[:-3]}-{grid}.h5",
        variables=["WP", "RV", "THETA", "PI"],
        bounds=xybounds,
        subset=True,
        subsetxy=True,
        coords=True,
    ) 

    dens = compute_dens(ds)
    w = ds.WP

    cond_mask = xr.open_dataarray(
        f"{outPath}/{run}/{grid}/cond_masks/{path}", 
        engine="h5netcdf",
        chunks="auto",
    ).chunk(ds.chunks)
    w_mask = xr.open_dataarray(
        f"{outPath}/{run}/{grid}/w_masks/{path}",
        engine="h5netcdf",
        chunks="auto",
    ).chunk(ds.chunks)

    features = pd.read_parquet(
        f"{outPath}/{run}/{grid}/combined_w_cond_segmented_tracks.pq"
    )
    features = features[features.time == time].reset_index(drop=True)

    alt = xr.ones_like(cond_mask) * cond_mask.ztn
    cond_mask = cond_mask.assign_coords(dz=("Z", dz))
    dz = xr.ones_like(cond_mask) * cond_mask.dz

    statistics = {}
    statistics["cloud_top_height"] = np.max
    statistics["cloud_base_height"] = np.min
    cout = get_statistics_from_mask(
        features, cond_mask, alt, statistic=statistics
    )

    statistics = {}
    statistics["updraft_top_height"] = np.max
    statistics["updraft_base_height"] = np.min
    wout = get_statistics_from_mask(features, w_mask, alt, statistic=statistics)

    out = pd.merge(left=cout, right=wout)

    statistics = {}
    statistics["condensate_volume"] = np.sum
    vout = get_statistics_from_mask(
        features, cond_mask, dz, statistic=statistics
    )
    vout["condensate_volume"] = vout["condensate_volume"] * dxy * dxy

    out = pd.merge(left=out, right=vout)

    statistics = {}
    statistics["updraft_volume"] = np.sum
    vout = get_statistics_from_mask(features, w_mask, dz, statistic=statistics)
    vout["updraft_volume"] = vout["updraft_volume"] * dxy * dxy

    out = pd.merge(left=out, right=vout)

    statistics = {}
    statistics["updraft_max"] = np.max
    statistics["updraft_mean"] = np.mean
    statistics["updraft_percentiles"] = (np.percentile, {"q": [50, 90, 95, 99]})
    wout = get_statistics_from_mask(features, w_mask, w, statistic=statistics)

    out = pd.merge(left=out, right=wout)

    cmf_mask = cond_mask.where(w >= 1)

    statistics = {}
    statistics["intcmf_max"] = np.max
    statistics["intcmf_mean"] = np.mean
    statistics["intcmf_total"] = np.sum
    statistics["intcmf_percentiles"] = (np.percentile, {"q": [50, 90, 95, 99]})
    cmfout = get_statistics_from_mask(
        features, cmf_mask, dens * w * dz * dxy * dxy, statistic=statistics
    )

    out = pd.merge(left=out, right=cmfout)

    statistics = {}
    statistics["cmf_volume"] = np.sum
    cmfout = get_statistics_from_mask(
        features, cmf_mask, dz, statistic=statistics
    )
    out = pd.merge(left=out, right=cmfout)
    out["cmf_volume"] = out.cmf_volume * dxy * dxy

    statistics = {}
    statistics["cmf_max"] = np.max
    statistics["cmf_mean"] = np.mean
    statistics["cmf_percentiles"] = (np.percentile, {"q": [50, 90, 95, 99]})

    cmfout = get_statistics_from_mask(
        features, cmf_mask, dens * w, statistic=statistics
    )

    out = pd.merge(left=out, right=cmfout)

    del w_mask
    del cond_mask
    del ds

    return out


for grid, batch_size in zip(grids, [5, 1]):
    for run in runs:
        print(run, grid)
        dataPath = f"/monsoon/MODEL/LES_MODEL_DATA/V1/{run}/{grid.capitalize()}/out_30s/"
        tobacPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}"
        savePath = f"{tobacPath}/qc_tracks_statistics.pq"

        if os.path.exists(
            f"{tobacPath}/combined_w_cond_segmented_tracks.pq") and (not os.path.exists(savePath)):
            paths = sorted(os.listdir(f"{tobacPath}/cond_masks"))

            ds = client.map(
                get_statistics,
                paths,
                run=run,
                grid=grid,  
                batch_size=batch_size
            )
            ds = client.gather(ds)

            ds = tobac.utils.combine_feature_dataframes(
                ds,
                renumber_features=False,
                sort_features_by="frame",
            )
            save_files(ds, savePath)

client.close()
cluster.close()