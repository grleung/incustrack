import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from scipy.ndimage import labeled_comprehension, maximum_position, sum_labels, maximum, mean, minimum
import dask
import dask.distributed as dd

client = dd.Client("downdraft:8786")
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
    if len(sub)>0:
        fts = sub.feature.unique()
        time = pd.to_datetime(sub.timestr.iloc[0])

        if grid=='g3':
            latlon=False
            coords=False
        else:
            latlon=True
            coords=True

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

        ds = ds.assign(COND=temp['COND'], DENS=temp['DENS'],PCPT=compute_pcp(ds))

        ds = ds[["COND", "PCPT", "WP", "DENS"]]

        cond_mask = xr.open_dataset(
            f"{tobacPath}/cond_masks-weaker/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
            engine="h5netcdf",
            chunks="auto",
        )

        w_mask = xr.open_dataset(
            f"{tobacPath}/w_masks-weaker/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
            engine="h5netcdf",
            chunks="auto",
        )

        pcp_mask = xr.open_dataset(
            f"{tobacPath}/pcp_masks-weaker/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
            engine="h5netcdf",
            chunks="auto",
        )

        cond_mask = cond_mask.assign(alt=(("ztn"), alt / 1000))
        cond_mask = cond_mask.assign(dz=(("ztn"), dz))
        
        
        shape = (cond_mask.segmentation_mask / cond_mask.segmentation_mask).fillna(1)
        cond_dz = cond_mask.dz* shape
        cond_alt = cond_mask.alt * shape
        
        cmf = cond_mask.dz * dxy * dxy * ds.DENS * ((ds.WP).where(ds.WP > 0))

        wp = ds.WP
        pcp = ds.PCPT


        sub["CTH"] = labeled_comprehension(
            cond_alt,
            cond_mask.segmentation_mask,
            fts,
            np.nanmax,
            np.float64,
            np.nan,
        )

        sub["CBH"] = labeled_comprehension(
            cond_alt,
            cond_mask.segmentation_mask,
            fts,
            np.nanmin,
            np.float64,
            np.nan,
        )

        

        sub["condensate_volume"] = sum_labels(
            cond_dz, cond_mask.segmentation_mask,
            fts,
        )* dxy* dxy/ (1000 * 1000 * 1000)

        sub["condensate_count"] = sum_labels(
            shape,
            cond_mask.segmentation_mask,
            fts,
        )

        print('cloud stats')

        sub["updraft_topalt"] = maximum(
            cond_alt,w_mask.segmentation_mask,
            fts,
        )

        sub["updraft_botalt"] = minimum(
            cond_alt,w_mask.segmentation_mask,
            fts,
        )

        sub["updraft_volume"] = sum_labels(
            cond_dz, w_mask.segmentation_mask,
            fts,
        ) * dxy* dxy/ (1000 * 1000 * 1000)

        sub["updraft_count"] = sum_labels(
            shape,
            w_mask.segmentation_mask,
            fts,)

        print('updraft stats')

        sub["precip_area"] = sum_labels(
            (pcp_mask.segmentation_mask / pcp_mask.segmentation_mask),
            pcp_mask.segmentation_mask,
            fts,
        )*dxy * dxy/ (1000 * 1000)

        print('pcp stats')

        
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

        print('cmf stats')

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

        sub["wmax_loc"] = maximum_position(
            wp,
            w_mask.segmentation_mask,
            fts
        )

        print('w stats')
        

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

        print('pcp2 stats')

        sub['w_footprint'] =  [dxy * dxy *get_footprint((w_mask.segmentation_mask==ft).data)/1000**2 for ft in fts]

        sub['cond_footprint'] = [dxy * dxy * get_footprint((cond_mask.segmentation_mask==ft).data)/1000**2 for ft in fts]

        print('footprint stats')

        sub['cmfmax_alt'] = alt[pd.DataFrame(sub.cmfmax_loc.to_list())[0]]/1000
        sub.loc[sub[sub.cmf_max.isnull()].index,'cmfmax_alt'] = np.nan
        sub['wmax_alt'] = alt[pd.DataFrame(sub.wmax_loc.to_list())[0]]/1000
        

        return sub
    else:
        print(sub)



runs = ['DRC1.1-R-V1' ]
grids = ['g1']

for grid in grids:
    for run in runs:
        latbounds = outbounds.loc[run].values
    
        print(run, grid)

        dxy=get_xy_spacing(grid)

        dataPath = f"/monsoon/MODEL/LES_MODEL_DATA/V1/{run}/G3/out_30s/"
        tobacPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}"

        out = pd.read_parquet(f"{tobacPath}/combined_segmented_tracks-weaker.pq")

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

        if grid=='g3':
            for i, frames in enumerate(np.array_split(sorted(out.frame.unique()), 13)):
                if not os.path.exists(f"{tobacPath}/combined_segmented_tracks_statistics_{str(i).zfill(2)}.pq"):
                    print(i,frames)
                    
                    x = client.map(
                        get_masked_statistics,
                        [out[out.frame==frame] for frame in frames],
                        grid=grid,
                        dxy=get_xy_spacing(grid),
                        latbounds=latbounds,
                        dataPath=dataPath,
                        tobacPath=tobacPath,
                    )
                    x = client.gather(x)

                    x = pd.concat(x)

                    x.to_parquet(f"{tobacPath}/combined_segmented_tracks_statistics_{str(i).zfill(2)}.pq")
                    print('saved', i)

        else:
            x = client.map(
                get_masked_statistics,
                [out[out.frame==frame] for frame in sorted(out.frame.unique())],
                grid=grid,
                dxy=get_xy_spacing(grid),
                latbounds=latbounds,
                dataPath=dataPath,
                tobacPath=tobacPath,
                batch_size=1
            )
            x = client.gather(x)

            out = pd.concat(x)

            out.to_parquet(f"{tobacPath}/combined_segmented_tracks_statistics-weaker.pq")
