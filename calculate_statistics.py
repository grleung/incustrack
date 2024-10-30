import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from scipy.ndimage import (
    labeled_comprehension,
    maximum_position,
    sum_labels,
    maximum,
    mean,
    minimum,
)
import dask
import dask.distributed as dd

client = dd.Client("updraft:9999")
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
    alt,dz,
    cp,
)


def get_footprint(arr):
    return arr.max(axis=(0)).sum().compute()


outbounds = pd.read_pickle(f"/tempest/gleung/incustrack/bounds.pkl")


def get_masked_statistics(sub, grid, dxy, latbounds, dataPath, tobacPath):
    if len(sub) > 0:
        fts = sub.feature.unique()
        time = pd.to_datetime(sub.timestr.iloc[0])

        if grid == "g3":
            latlon = False
            coords = False
        else:
            latlon = True
            coords = True

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
            latlon=latlon,
            coords=coords,
        )

        temp = compute_cond(ds, return_dens=True)

        ds = ds.assign(
            COND=temp["COND"], DENS=temp["DENS"], PCPT=compute_pcp(ds)
        )

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

        #cond_mask = cond_mask.assign(alt=(("ztn"), alt / 1000))
        #cond_mask = cond_mask.assign(dz=(("ztn"), dz))
        cond_mask=cond_mask.rename_dims({'dim_0':'Z'})
        cond_mask = cond_mask.assign_coords(dz = ('Z',dz))

        shape = (
            cond_mask.segmentation_mask / cond_mask.segmentation_mask
        ).fillna(1)
        cond_dz = cond_mask.dz * shape
        cond_alt = cond_mask.ztn * shape

        cmf_mask = cond_mask.dz * dxy * dxy * (shape.where(ds.WP > 0))
        cmf = (
            cmf_mask * ds.DENS * ds.WP
        )  # cond_mask.dz * dxy * dxy * ds.DENS * ((ds.WP).where(ds.WP > 0))

        wp = ds.WP
        pcp = ds.PCPT

        sub["CTH"] = labeled_comprehension(
            cond_alt/1000,
            cond_mask.segmentation_mask,
            fts,
            np.nanmax,
            np.float64,
            np.nan,
        )

        sub["CBH"] = labeled_comprehension(
            cond_alt/1000,
            cond_mask.segmentation_mask,
            fts,
            np.nanmin,
            np.float64,
            np.nan,
        )

        sub["condensate_volume"] = (
            sum_labels(
                cond_dz,
                cond_mask.segmentation_mask,
                fts,
            )
            * dxy
            * dxy
            / (1000 * 1000 * 1000)
        )

        sub["condensate_count"] = sum_labels(
            shape,
            cond_mask.segmentation_mask,
            fts,
        )

        print("cloud stats")

        sub["updraft_topalt"] = maximum(
            cond_alt/1000,
            w_mask.segmentation_mask,
            fts,
        )

        sub["updraft_botalt"] = minimum(
            cond_alt/1000,
            w_mask.segmentation_mask,
            fts,
        )

        sub["updraft_volume"] = (
            sum_labels(
                cond_dz,
                w_mask.segmentation_mask,
                fts,
            )
            * dxy
            * dxy
            / (1000 * 1000 * 1000)
        )

        sub["updraft_count"] = sum_labels(
            shape,
            w_mask.segmentation_mask,
            fts,
        )

        print("updraft stats")

        sub["pcp_area"] = (
            sum_labels(
                (pcp_mask.segmentation_mask / pcp_mask.segmentation_mask),
                pcp_mask.segmentation_mask,
                fts,
            )
            * dxy
            * dxy
            / (1000 * 1000)
        )

        print("pcp stats")

        sub["cmf_volume"] = labeled_comprehension(
            cmf_mask,
            cond_mask.segmentation_mask,
            fts,
            np.nansum,
            np.float64,
            np.nan,
        )

        sub["dens_weighted_w_mean"] = labeled_comprehension(
            ds.WP * ds.DENS,
            cond_mask.segmentation_mask,
            fts,
            np.nanmean,
            np.float64,
            np.nan,
        )

        sub["cmf_total"] = labeled_comprehension(
            cmf, cond_mask.segmentation_mask, fts, np.nansum, np.float64, np.nan
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
            cmf.fillna(0), cond_mask.segmentation_mask, fts
        )

        print("cmf stats")

        sub["w_mean"] = mean(
            wp,
            w_mask.segmentation_mask,
            fts,
        )

        sub["w_max"] = maximum(
            wp,
            w_mask.segmentation_mask,
            fts,
        )

        sub["wmax_loc"] = maximum_position(wp, w_mask.segmentation_mask, fts)

        print("w stats")

        sub["pcp_mean"] = mean(
            pcp,
            pcp_mask.segmentation_mask,
            fts,
        )

        sub["pcp_max"] = maximum(
            pcp,
            pcp_mask.segmentation_mask,
            fts,
        )

        sub["pcp_total"] = sub.pcp_mean * sub.pcp_area

        print("pcp2 stats")

        """if grid=='g1':
            sub['w_footprint'] =  [dxy * dxy *get_footprint((w_mask.segmentation_mask==ft).data)/1000**2 for ft in fts]
            sub['condensate_count'] = [dxy * dxy * get_footprint((cond_mask.segmentation_mask==ft).data)/1000**2 for ft in fts]

            print('footprint stats')
        """
        sub["cmfmax_alt"] = (
            alt[pd.DataFrame(sub.cmfmax_loc.to_list())[0]] / 1000
        )
        sub.loc[sub[sub.cmf_max.isnull()].index, "cmfmax_alt"] = np.nan
        sub["wmax_alt"] = alt[pd.DataFrame(sub.wmax_loc.to_list())[0]] / 1000

        print('done!')

        return sub
    else:
        print(sub)


runs = ['DRC1.1-R-V1']  
grids = ["g3"]

for grid in grids:
    for run in runs:
        latbounds = outbounds.loc[run].values

        print(run, grid)

        dxy = get_xy_spacing(grid)

        dataPath = f"/monsoon/MODEL/LES_MODEL_DATA/V1/{run}/G3/out_30s/"
        tobacPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1-temp/{run}/{grid}"

        # need to take feature numbering from tracks parquet file
        tracks = pd.read_parquet(f"{tobacPath}/w_tracks.pq")
        pcp = pd.read_parquet(
            f"{tobacPath}/combined_w_cond_pcp_segmented_tracks.pq"
        )

        out = pd.merge(
            left=tracks,
            right=pcp,
            left_on=["cell", "frame"],
            right_on=["cell", "frame"],
            suffixes=[None, "_drop"],
        )
        out = out[[c for c in out.columns if not c.endswith("_drop")]]
        out["pcp_ncells"] = out["ncells"].copy()


        for i, frames in enumerate(
            np.array_split(sorted(out.frame.unique()), 11)
        ):
            if not os.path.exists(
                f"{tobacPath}/qc_feature_statistics_{str(i).zfill(2)}.pq"
                ):

                        
                x = client.map(
                    get_masked_statistics,
                    [out[out.frame == frame] for frame in frames],
                    grid=grid,
                    dxy=get_xy_spacing(grid),
                    latbounds=latbounds,
                    dataPath=dataPath,
                    tobacPath=tobacPath,
                )
                x = client.gather(x)

                x = pd.concat(x)

                x.to_parquet(
                    f"{tobacPath}/qc_feature_statistics_{str(i).zfill(2)}.pq"
                )

