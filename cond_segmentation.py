# Import some shared libraries
import os
import dask.distributed as dd
import dask
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import glob

client = dd.Client("updraft:8786")
client.upload_file("shared_functions.py")

from shared_functions import (
    get_rams_output,
    get_xy_spacing,
    subset_data,
    combine_tobac_list,
    rams_dims_lite,
    all_var,
    save_files,
    compute_cond,
    p00,
    rd,
    cp,
)

# Define the paths to INCUS data and where to save output
ver = "V1"  # version of INCUS simulation dataset
modelPath = f"/monsoon/MODEL/LES_MODEL_DATA/{ver}/"
outPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/{ver}/"
runs = ["WPO1.1-R-V1"]  # which model runs to process
grids = ["g3"]

# parameters for segmentation
params = {}
params["method"] = "watershed"
params["threshold"] = 1.0e-5  # kg/m3 mixing ratio
params["seed_3D_flag"] = "box"
params["vertical_coord"] = "ztn"

# separately I created a pkl file that contains the min/max lat/lon for each of the simulations
# having this as separate dataframe saves on some computational cost from re-calculating this
# in every script
outbounds = pd.read_pickle(f"/tempest/gleung/incustrack/bounds.pkl")

# loop through each of the runs

for grid in grids:
    for run in runs:
        print(grid, run)

        dataPath = f"{modelPath}/{run}/G3/out_30s/"

        latbounds = outbounds.loc[run].values

        # list of all timesteps where lite files are found in relevant folder
        all_paths = [
            p.split("/")[-1][:-6]
            for p in sorted(glob.glob(f"{dataPath}/a-L-*-g3.h5"))
        ]

        trackPath = f"{outPath}/{run}/{grid}/w_tracks.pq"
        tracks = pd.read_parquet(trackPath)

        if grid == "g3":
            n = 40
            batch_size = 1
            all_paths = enumerate(np.array_split(all_paths, n))
        elif grid == "g2":
            n = 5
            batch_size = 1
            all_paths = enumerate(np.array_split(all_paths, n))
        else:
            n = 1
            batch_size = 20
            all_paths = enumerate([all_paths])

        for i, paths in all_paths:
            times = [pd.to_datetime(p.split("/")[-1][4:]) for p in paths]

            savemaskPath = f"{outPath}/{run}/{grid}/cond_masks/"

            if grid != "g1":
                batch_size = 1
                savedfPath = (
                    f"{outPath}/{run}/{grid}/cond_seg_{str(i).zfill(2)}.pq"
                )

            else:
                batch_size = 20
                savedfPath = f"{outPath}/{run}/{grid}/cond_seg.pq"

            if not os.path.exists(savedfPath):

                dxy = get_xy_spacing(grid)

                ds = client.map(
                    get_rams_output,
                    [f"{dataPath}/{p}-{grid}.h5" for p in paths],
                    variables=["RCP", "RSP", "RPP", "PI", "THETA", "RV"],
                    latbounds=latbounds,
                    latlon=True,
                    coords=True,
                    batch_size=batch_size,
                )

                ds = client.map(compute_cond, ds, batch_size=batch_size)

                ds = client.map(
                    xr.DataArray.expand_dims,
                    ds,
                    [{"time": [t]} for t in times],
                    batch_size=batch_size,
                )

                ds = client.map(
                    xr.DataArray.to_iris,
                    ds,
                    batch_size=batch_size,
                )

                out = client.map(
                    tobac.segmentation.segmentation,
                    [tracks[tracks.time == t] for t in times],
                    ds,
                    dxy=dxy,
                    **params,
                    batch_size=batch_size,
                )

                out = client.gather(out)

                all_segments = [o[1] for o in out]
                all_masks = [o[0] for o in out]

                # just make sure all directories exist
                if not os.path.isdir(f"{outPath}/{run}"):
                    os.mkdir(f"{outPath}/{run}")

                if not os.path.isdir(f"{outPath}/{run}/{grid}"):
                    os.mkdir(f"{outPath}/{run}/{grid}")

                # once loop is finished, concatenate all the figures
                # then save it to a parquet file
                all_segments = combine_tobac_list(all_segments)
                save_files(all_segments, savedfPath)

                if not os.path.isdir(savemaskPath):
                    os.mkdir(savemaskPath)

                for m, p in zip(all_masks, paths):
                    ds = xr.DataArray.from_iris(m)
                    ds.to_netcdf(
                        f"{savemaskPath}/{p}.h5",
                        engine="h5netcdf",
                        encoding={
                            "segmentation_mask": {"zlib": True, "complevel": 9}
                        },
                    )
