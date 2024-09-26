import os 
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt

runs = ['AUS1.1-R-V1']#'PHI1.1-R-V1','PHI2.1-R-V1','DRC1.1-R-V1']
grids = ['g1','g2']


for run in runs:
    for grid in grids:
        print(run,grid)
        dataPath = f'/monsoon/MODEL/LES_MODEL_DATA/V1/{run}/{grid.capitalize()}/out_30s/'
        tobacPath = f'/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}'

        w = pd.read_parquet(f"{tobacPath}/w_seg.pq")
        pcp = pd.read_parquet(f"{tobacPath}/pcp_seg.pq")
        cond = pd.read_parquet(f"{tobacPath}/cond_seg.pq")
        tracks = pd.read_parquet(f"{tobacPath}/w_tracks.pq")

        out = pd.merge(left=tracks,right=cond,left_on=['cell','frame'],right_on=['cell','frame'],suffixes=[None,'_drop'])
        out = out[[c for c in out.columns if not c.endswith('_drop')]]
        out['cond_ncells'] = out['ncells'].copy()

        out = pd.merge(left=out,right=w,left_on=['cell','frame'],right_on=['cell','frame'],suffixes=[None,'_drop'])
        out['w_ncells'] = out['ncells_drop'].copy()
        out = out[[c for c in out.columns if not c.endswith('_drop')]]

        out = pd.merge(left=out,right=pcp,left_on=['cell','frame'],right_on=['cell','frame'],suffixes=[None,'_drop'])
        out['pcp_ncells'] = out['ncells_drop'].copy()
        out = out[[c for c in out.columns if not c.endswith('_drop')]]

        out = out.drop(columns=['ncells'])

        out['lifetime'] = out.groupby('cell').time_cell.transform('max')/dt.timedelta(minutes=1)

        out.to_parquet(f"{tobacPath}/combined_segmented_tracks.pq",engine='pyarrow')