import os 
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt

runs = ['DRC1.1-R-V1','PHI1.1-R-V1','PHI2.1-R-V1','ARG1.2-R-V1','AUS1.1-R-V1','USA1.1-R-V1','WPO1.1-R-V1']
grids =['g1','g2','g3']

for run in runs:
    for grid in grids:
        print(run,grid)
        dataPath = f'/monsoon/MODEL/LES_MODEL_DATA/V1/{run}/{grid.capitalize()}/out_30s/'
        tobacPath = f'/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}'


        if (not os.path.exists(f"{tobacPath}/combined_w_cond_segmented_tracks.pq")) and (os.path.exists(f"{tobacPath}/cond_seg.pq"))  and (os.path.exists(f"{tobacPath}/w_seg.pq")):
            print('merging')
            w = pd.read_parquet(f"{tobacPath}/w_seg.pq")
            cond = pd.read_parquet(f"{tobacPath}/cond_seg.pq")
            tracks = pd.read_parquet(f"{tobacPath}/w_tracks.pq")


            out = pd.merge(left=tracks,right=cond,left_on=['cell','frame'],right_on=['cell','frame'],suffixes=[None,'_drop'])
            out = out[[c for c in out.columns if not c.endswith('_drop')]]
            out['cond_ncells'] = out['ncells'].copy()

            out = pd.merge(left=out,right=w,left_on=['cell','frame'],right_on=['cell','frame'],suffixes=[None,'_drop'])
            out['w_ncells'] = out['ncells_drop'].copy()
            out = out[[c for c in out.columns if not c.endswith('_drop')]]

            out = out.drop(columns=['ncells'])

            # removing features where there is no w or condensate tracked throughout 
            # entire life before doing precipitation segmentation to improve segmentation
            out['cellmax_wncells'] = out.groupby('cell').w_ncells.transform('max')
            out['cellmax_condncells'] = out.groupby('cell').cond_ncells.transform('max')

            print(f'started with {len(out)} features')
            out = out[(out.cellmax_wncells>0) & (out.cellmax_condncells>0)]
            print(f'ended with {len(out)} features')

            out['lifetime'] = out.groupby('cell').time_cell.transform('max')/dt.timedelta(minutes=1)

            out.to_parquet(f"{tobacPath}/combined_w_cond_segmented_tracks.pq",engine='pyarrow')
