# This script will take a glob as an argument in command line and merge all associated parquet files into one
# To run, do: python merge_parquets.py input_glob output_path renum_flag
# e.g.: "python merge_parquets.py '/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/DRC1.1-R-V1/g3/cond_seg_*.pq' '/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/DRC1.1-R-V1/g3/cond_seg.pq'" False


import pandas as pd
import os
import numpy as np
import sys
import glob
from shared_functions import save_files
from tobac.utils import combine_feature_dataframes

# read the glob passed in and find relevant files
dataPaths = sorted(glob.glob(sys.argv[1]))
# output path
savePath = sys.argv[2]

# flag for renumbering features or not
renum_flag = sys.argv[3]

all_df = []
for p in dataPaths:
    df = pd.read_parquet(p, engine="pyarrow")

    all_df.append(df)
    print(p, len(df))


if "qc_feature" in savePath:
    all_df = pd.concat(all_df).sort_values("feature")
    all_df = all_df.drop_duplicates(
        subset=["vdim", "hdim_1", "hdim_2", "feature"]
    )

else:
    # use tobac tool so that frame numbering is correct
    all_df = combine_feature_dataframes(
        all_df, renumber_features=renum_flag, sort_features_by="frame"
    )

save_files(all_df, savePath)

print(
    f"Combined {len(dataPaths)} files to get a total of {len(all_df)} features. File is saved in {savePath}."
)
