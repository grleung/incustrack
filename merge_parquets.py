# This script will take a glob as an argument in command line and merge all associated parquet files into one
# To run, do: python merge_parquets.py input_glob output_path
# e.g.: "python merge_parquets.py '/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/DRC1.1-R-V1/g3/cond_seg_*.pq' '/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/DRC1.1-R-V1/g3/cond_seg.pq'" 

import pandas as pd
import os
import numpy as np
import sys
import glob
from shared_functions import combine_tobac_list, save_files

# read the glob passed in and find relevant files
dataPaths = sorted(glob.glob(sys.argv[1]))
#output path
savePath = sys.argv[2]

all_df = []
for p in dataPaths:
    df = pd.read_parquet(p,engine='pyarrow')

    all_df.append(df)
    print(p, len(df))

# use tobac tool so that frame numbering is correct
all_df = combine_tobac_list(all_df)

save_files(all_df, savePath)

print(f'Combined {len(dataPaths)} files to get a total of {len(all_df)} features. File is saved in {savePath}.')