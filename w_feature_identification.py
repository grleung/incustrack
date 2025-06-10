# This is the first script to run. Here, we use tobac to identify updraft
# features that meet the given vertical velocity thresholds.

# inputs: list of model runs, RAMS model output
# output: tobac output dataframe of features

# This script is parallelized with dask distributed. To run from command line
# run "dask scheduler &" then "dask workers tcp::/scheduler:port --nworkers NW --nthreads NT --memory-limit "MEM GiB"
# I usually use NW = 30, NT = 1, MEM = 30 GiB on downdraft, but haven't really experimented with this.
# then run "python w_feature_identification.py"
# Make sure that the client below is the same as the client address of your scheduler.
# Pretty sure this is the default and should work, but I may be wrong.

# For questions, contact Bee (gabrielle.leung@colostate.edu)

# Import some shared libraries
import os
import dask.distributed as dd
from dask_jobqueue import SLURMCluster
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import glob

# spin up SLURM cluster
cluster = SLURMCluster(cores=90,
                       memory='600GB',
                       account='incus',
                       walltime='08:00:00',
                       scheduler_options={'dashboard_address':":10101"},
                       job_extra_directives=['--partition=all',
                                             '--job-name=tobac-feature-detection'])
cluster.scale(jobs=2)

# change this address depending on your scheduler address
client = dd.Client(cluster)
client.upload_file("shared_functions.py")
from shared_functions import (
    get_rams_output,
    get_xy_spacing,
    combine_tobac_list,
    save_files,
)

# Define the paths to INCUS data and where to save output
ver = "V1"  # version of INCUS simulation dataset
modelPath = f"/monsoon/MODEL/LES_MODEL_DATA/{ver}/"
outPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/{ver}/"

runs = ['USA1.1-R-V1', 'AUS1.1-R-V1', 'DRC1.1-R-V1', 
        'BRA2.1-R-V1', 'ARG1.1-R-V1', 'SIO1.1-R-V1',
        'USA3.1-R-V1', 'WPO1.1-R-V1', 'PHI2.1-R-V1', 
        'ARG1.2-R-V1', 'WPO1.1-RPR-V1', 'PHI1.1-R-V1', 
        'SAU1.1-R-V1', 'BRA1.1-R-V1', 'DRC1.1-RCR-V1'] # which model runs to process
grids = ["g2"]

# separately I created a pkl file that contains the min/max lat/lon for each of the simulations
# having this as separate dataframe saves on some computational cost from re-calculating this
# in every script
outbounds = pd.read_pickle(f"/tempest/gleung/incustrack/bounds.pkl")

# tobac feature identification parameters
# see tobac documentation for more detailed description
params = {}
params["position_threshold"] = "weighted_diff"
params["sigma_threshold"] = 1
params["n_erosion_threshold"] = 0
# this is ~4 points in each direction; we had some discussion about changing
# this for different grid spacings, but as of 2024-08-13 this makes more sense to me (Bee)
params["n_min_threshold"] = 64
params["target"] = "maximum"
# threshold in m/s is (1, 2, 4, 6, ..., 50)
params["threshold"] = np.append([1.0], np.arange(2.0, 52.0, 2.0))

# loop through each of the runs
for run in runs:
    dataPath = f"{modelPath}/{run}/G3/out_30s/"

    # list of all timesteps where lite files are found in relevant folder
    all_paths = [
        p.split("/")[-1][:-6]
        for p in sorted(glob.glob(f"{dataPath}/a-L-*-g3.h5"))
    ]

    print(len(all_paths))

    latbounds = outbounds.loc[run].values

    for grid in grids:

        # For some of the g3 domains, I had some issues with too much data being loaded into memory at once,
        # so I split up feature detection into multiple subsets
        if grid == "g3":
            n_split = 16
        else:
            n_split = 1

        for i, paths in enumerate(np.array_split(all_paths, n_split)):
            if grid == "g3":
                savedfPath = (
                    f"{outPath}/{run}/{grid}/w_features_{str(i).zfill(2)}.pq"
                )
            else:
                savedfPath = f"{outPath}/{run}/{grid}/w_features.pq"

            if not os.path.exists(savedfPath):

                # batch size is how may batches to submit the tasks in the list [paths] to scheduler
                if grid == "g3":
                    batch_size = 1
                else:
                    batch_size = 20

                dxy = get_xy_spacing(grid)

                # prep data for feeding to tobac
                ds = client.map(
                    get_rams_output,
                    [f"{dataPath}/{p}-{grid}.h5" for p in paths],
                    variables=["WP"],
                    latbounds=latbounds,
                    latlon=True,
                    coords=True,
                    batch_size=batch_size,
                )

                ds = client.map(
                    xr.DataArray.expand_dims,
                    ds,
                    [
                        {"time": [pd.to_datetime(p.split("/")[-1][4:])]}
                        for p in paths
                    ],
                    batch_size=batch_size,
                )

                # actual tobac run
                feats = client.map(
                    tobac.feature_detection_multithreshold,
                    ds,
                    dxy=dxy,
                    vertical_coord="ztn",
                    **params,
                    batch_size=batch_size,
                )

                # take all the features from tobac run
                all_features = client.gather(feats)

                # once loop is finished, concatenate all the figures
                # then save it to a parquet file
                all_features = combine_tobac_list(all_features)

                # just make sure all directories exist
                if not os.path.isdir(f"{outPath}/{run}"):
                    os.mkdir(f"{outPath}/{run}")

                if not os.path.isdir(f"{outPath}/{run}/{grid}"):
                    os.mkdir(f"{outPath}/{run}/{grid}")

                save_files(all_features, savedfPath)

cluster.close()
client.close()