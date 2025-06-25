import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt

runs = [
    "ARG1.1-R-V1",
    "ARG1.2-R-V1",
    "AUS1.1-R-V1",
    "BRA1.1-R-V1",
    "BRA2.1-R-V1",
    "DRC1.1-R-V1",
    "DRC1.1-RCR-V1",
    "PHI1.1-R-V1",
    "PHI2.1-R-V1",
    "SAU1.1-R-V1",
    "SIO1.1-R-V1",
    "USA1.1-R-V1",
    "USA3.1-R-V1",
    "WPO1.1-R-V1",
    "WPO1.1-RPR-V1",
]
grids = ["g1", "g2"]

for run in runs:
    for grid in grids:
        print(run, grid)
        dataPath = f"/monsoon/MODEL/LES_MODEL_DATA/V1/{run}/{grid.capitalize()}/out_30s/"
        tobacPath = (
            f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}"
        )

        if (
            (
                not os.path.exists(
                    f"{tobacPath}/combined_w_cond_segmented_tracks.pq"
                )
            )
            and (os.path.exists(f"{tobacPath}/cond_seg.pq"))
            and (os.path.exists(f"{tobacPath}/w_seg.pq"))
        ):
            print("merging")
            tracks = pd.read_parquet(
                f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}/w_tracks.pq"
            )
            w = pd.read_parquet(
                f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}/w_seg.pq"
            )
            cond = pd.read_parquet(
                f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}/cond_seg.pq"
            )

            if (
                sorted(w.feature.values) == sorted(tracks.feature.values)
            ) != True:
                w["feature"] = w.set_index(["frame", "cell"]).index.map(
                    tracks.set_index(["frame", "cell"]).feature
                )

                print("renumbered w features")

            if (
                sorted(cond.feature.values) == sorted(tracks.feature.values)
            ) != True:
                cond["feature"] = cond.set_index(["frame", "cell"]).index.map(
                    tracks.set_index(["frame", "cell"]).feature
                )

                print("renumbered cond features")

            w = w.set_index("feature")
            cond = cond.set_index("feature")

            tracks["ncells_w"] = tracks.feature.map(w.ncells)
            tracks["ncells_cond"] = tracks.feature.map(cond.ncells)

            tracks["lifetime"] = tracks.groupby("cell").time_cell.transform(
                "max"
            ) / dt.timedelta(minutes=1)
            tracks["frac_lifetime"] = (
                tracks.time_cell / dt.timedelta(minutes=1)
            ) / tracks.lifetime

            tracks["cellmax_nw"] = tracks.groupby("cell").ncells_w.transform(
                "max"
            )
            tracks["cellmax_ncond"] = tracks.groupby(
                "cell"
            ).ncells_cond.transform("max")
            print(len(tracks))

            thresh = 64
            tracks = tracks[
                (tracks.cellmax_nw >= thresh) & (tracks.cellmax_ncond >= thresh)
            ]

            tracks.to_parquet(
                f"{tobacPath}/combined_w_cond_segmented_tracks.pq",
                engine="pyarrow",
            )

        print(run, grid)
