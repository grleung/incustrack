# This is the first script to run. Here, we use tobac to identify updraft
# features that meet the given vertical velocity thresholds.

# inputs: list of model runs, RAMS model output
# output: tobac output dataframe of features

# To run this script, in command line enter "jug execute w_feature_identification.py &"
# repeated n times (where n is the number of processes you want to run).
# This script will process each timestep separately then stitch them together.

# For questions, contact Bee (gabrielle.leung@colostate.edu)

# Import some shared libraries
import os
from jug import TaskGenerator
import numpy as np
import pandas as pd
import xarray as xr
from shared_functions import (
    get_rams_output,
    get_xy_spacing,
    subset_data,
    combine_tobac_list,
    save_df,
)


# Define the paths to INCUS data and where to save output
ver = "V1"  # version of INCUS simulation dataset
modelPath = f"/monsoon/MODEL/LES_MODEL_DATA/{ver}/"
outPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/{ver}/"
runs = ["DRC1.1-R-V1"]  # which model runs to process

"""
# This will make a list of runs that has all variabels in modelPath where 30-s output exists
runs = [
    r
    for r in os.listdir(modelPath)
    if (os.path.exists(f"{modelPath}/{r}/G3/out_30s"))
]

runs = [
    r
    for r in sorted(runs)
    if (
        (
            len(
                os.listdir(
                    f"{modelPath}/{r}/G3/out_30s"
                )
            )
            > 0
        )
        & ("old" not in r)
    )
]

runs = [r for r in runs if ("-R-" in r)] # subset only base aerosol
"""

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


@TaskGenerator
def get_w_features(dataPath, p, grid, latbounds):
    # parse the file path to get datetime since format is always the same in RAMS
    time = pd.to_datetime(p[4:])
    p = f"{dataPath}/{p}-{grid}.h5"

    # assign dxy (horizontal grid spacing) based on grid
    dxy = get_xy_spacing(grid)

    # read in the RAMS data and assign coordinates in format tobac expects
    ds = get_rams_output(p, ["WP"], latlon=True, coords=True)
    ds = ds.expand_dims(dim="time")
    ds["time"] = [time]

    # subset data based on lat/lon bounds
    # this is to make sure that we're using the same spatial location for all the grids
    ds = subset_data(ds, latbounds)

    # right now, tobac doesn't have xarray support, so convert to an iris cube
    iris_df = ds["WP"].to_iris()

    # call tobac feature detection
    Features = tobac.feature_detection_multithreshold(
        iris_df, dxy, vertical_coord="ztn", **params
    )

    return Features


# separately I created a pkl file that contains the min/max lat/lon for each of the simulations
# having this as separate dataframe saves on some computational cost from re-calculating this
# in every script
outbounds = pd.read_pickle(f"/tempest/gleung/incustrack/bounds.pkl")

# actual loop
for run in runs:
    dataPath = f"{modelPath}/{run}/G3/out_30s/"

    # list of all timesteps where lite files are found in relevant folder
    paths = [
        p[:-6]
        for p in sorted(os.listdir(dataPath))
        if p.startswith("a-L") & p.endswith("-g3.h5")
    ]

    latbounds = outbounds.loc[run].values

    for grid in ["g1", "g2", "g3"]:
        # define path to save variable in
        savePath = f"{outPath}/{run}/{grid}/w_features.pkl"

        # just make sure all directories exist
        if not os.path.isdir(f"{outPath}/{run}"):
            os.mkdir(f"{outPath}/{run}")

        if not os.path.isdir(f"{outPath}/{run}/{grid}"):
            os.mkdir(f"{outPath}/{run}/{grid}")

        # loop over timesteps and append each output dataframe to list
        all_features = list()
        for p in paths:
            all_features.append(get_w_features(dataPath, p, grid, latbounds))

        # once loop is finished, concatenate all the figures and
        # save dataframe as a pickle in savePath
        combined_df = combine_tobac_list(all_features)
        save_df(combined_df, savePath)
