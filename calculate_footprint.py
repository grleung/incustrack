import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from scipy.ndimage import labeled_comprehension, find_objects
import dask
import dask.distributed as dd
import glob

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

def calculate_footprint(ft, loc, arr):
    mask = arr.sel(ztn=loc[0],Y=loc[1],X=loc[2])==ft
    return(get_footprint(mask.data).compute())

    
def get_footprint(arr):
    return(arr.max(axis=(0)).sum())

runs = ['AUS1.1-R-V1']
grid = 'g3'
dxy=100

for run in runs:
    dataPath = f"/monsoon/MODEL/LES_MODEL_DATA/V1/{run}/G3/out_30s/"
    tobacPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}"

    out = pd.read_parquet(f"{tobacPath}/combined_segmented_tracks_statistics.pq")
    frames = sorted(out.frame.unique())
    
    cond_paths = sorted(glob.glob(f"{tobacPath}/cond_masks/a-L-*.h5"))
    w_paths = sorted(glob.glob(f"{tobacPath}/w_masks/a-L-*.h5"))

    print(len(frames), len(cond_paths))

    out_df = []
    print('go')

    for frame, cond_path, w_path in zip(frames, cond_paths, w_paths):
        sub = out[out.frame==frame]
        fts = sub.feature.unique()

        cond_mask = xr.open_dataset(
            cond_path,
            engine="h5netcdf",
            chunks="auto",
            )

        print('read cond',frame)

        locs = find_objects(cond_mask.segmentation_mask)
        locs = [l for l in locs if l!=None]
        locs = pd.DataFrame(locs,index=fts[sub['condensate_volume']>0])
        locs = [x[1] for x in locs.iterrows()]

        print('start cond',frame)
        x = client.map(calculate_footprint,fts[sub['condensate_volume']>0], locs, arr=cond_mask.segmentation_mask)
        x = client.gather(x)
        x = pd.DataFrame(x, index=fts[sub['condensate_volume']>0])
        
        sub['cond_footprint'] = sub.feature.map(x[0]) * (dxy*dxy/(1000**2))
        print('cond done',frame)

        w_mask = xr.open_dataset(
            w_path,
            engine="h5netcdf",
            chunks="auto",
            )
        print('read w',frame)

        locs = find_objects(w_mask.segmentation_mask)
        locs = [l for l in locs if l!=None]
        locs = pd.DataFrame(locs,index=fts[sub['updraft_volume']>0])
        locs = [x[1] for x in locs.iterrows()]

        x = client.map(calculate_footprint, fts[sub['updraft_volume']>0],locs,arr=w_mask.segmentation_mask)
        x = client.gather(x)
        x = pd.DataFrame(x, index=fts[sub['updraft_volume']>0])

        sub['w_footprint'] = sub.feature.map(x[0]) * (dxy*dxy/(1000**2))
        print('w done',frame)

        out_df.append(sub)

    out_df = pd.concat(out_df)

    print(out_df)
    
    out_df.to_parquet(f"{tobacPath}/combined_segmented_tracks_statistics.pq")
    print('saved!')
                    
    


