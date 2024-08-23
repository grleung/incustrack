# This is the second script to run. Here, we use tobac to connect features in time
# to form cell tracks then subset those tracks which do not go within 9.6km of domain edges
# and which last longer than 2 minutes

# inputs: expected storm motion velocity from Peter/IT [to follow]
#         tobac output dataframe of features (from w_feature_identification)
# output: tobac output dataframe of tracked cells

# To run this script, just do "python w_cell_tracking.py" in command line.

# For questions, contact Bee (gabrielle.leung@colostate.edu)

# Import some shared libraries
import os
import dask.distributed as dd
import dask
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import tobac


from shared_functions import get_xy_spacing, save_files

# Define the paths to INCUS data and where to save output
ver = "V1"  # version of INCUS simulation dataset
modelPath = f"/monsoon/MODEL/LES_MODEL_DATA/{ver}/"
outPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/{ver}/"
runs = [
    "DRC1.1-R-V1",
]  # which model runs to process

# separately I created a pkl file that contains the number of x,y points for each of the simulations
# having this as separate dataframe saves on some computational cost from re-calculating this
# in every script
outx = pd.read_pickle(f"/tempest/gleung/incustrack/nx.pkl")
outy = pd.read_pickle(f"/tempest/gleung/incustrack/ny.pkl")

# tobac tracking parameters
# see tobac documentation for more detailed description of each parameter
params = {}
params["extrapolate"] = 0
params["order"] = 1
params["memory"] = 0
params["time_cell_min"] = 30  # in seconds
params["method_linking"] = "predict"
params["adaptive_step"] = 0.75
params["adaptive_stop"] = 1.0

# actual loop
for run in runs:
    dataPath = f"{modelPath}/{run}/G3/out_30s/"

    for grid in ["g3"]:
        featPath = f"{outPath}/{run}/{grid}/w_features.pq"
        savePath = f"{outPath}/{run}/{grid}/w_tracks_1850m.pq"

        # assign dxy (horizontal grid spacing) based on grid
        dxy = get_xy_spacing(grid)

        # once we are running with separate predicted track speed for each simulation, that goes here
        # for now use just the same one
        params["d_max"] = 1850

        # check if there is already a pickle file for w tracks so we don't re-do this
        # and check to make sure pickle file with features already exists
        # if (not os.path.exists(savePath)) & (os.path.exists(featPath)):
        if True:
            nx = outx.loc[run, grid]
            ny = outy.loc[
                run,
                grid,
            ]
            min_h1, max_h1 = 9600 / dxy, ny - 9600 / dxy
            min_h2, max_h2 = 9600 / dxy, nx - 9600 / dxy

            features = pd.read_parquet(featPath)

            # Perform linking and save results:
            tracks = tobac.linking_trackpy(
                features,
                None,
                dt=30,  # time in seconds separating each frame
                dxy=dxy,
                vertical_coord="ztn",
                **params,
            )

            # remove any features that don't belong to a track (in tobac these are labelled -1)
            out_tracks = tracks[tracks.cell != -1]

            # first remove tracks that last less than 2 minutes
            out_tracks = out_tracks[
                out_tracks.groupby("cell")["time_cell"].transform("max")
                > dt.timedelta(minutes=2)
            ]

            # second filter out cells near the edges of the nested grid

            # make sure lat lon are float
            out_tracks["lat"] = out_tracks["lat"].astype("float")
            out_tracks["lon"] = out_tracks["lon"].astype("float")

            # select only cells where the cell is never outside the max/min horizontal locations
            cells = out_tracks[["cell", "hdim_1", "hdim_2"]].groupby("cell")

            out_tracks = out_tracks[
                (
                    (cells.hdim_1.transform("min") > min_h1)
                    & (cells.hdim_2.transform("min") > min_h2)
                )
                & (
                    (cells.hdim_1.transform("max") < max_h1)
                    & (cells.hdim_2.transform("max") < max_h2)
                )
            ]

            # just make sure all directories exist
            if not os.path.isdir(f"{outPath}/{run}"):
                os.mkdir(f"{outPath}/{run}")

            if not os.path.isdir(f"{outPath}/{run}/{grid}"):
                os.mkdir(f"{outPath}/{run}/{grid}")

            save_files(out_tracks, savePath)
