import numpy as np
import pandas as pd
import os
from pathlib import Path
import shutil

for i in range(1,4):
    path = f"selected_batch_{i}.csv"
    print(f"Current processing file: {path}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    
    folder_paths = df["folder_path"]
    image_names = df["image_name"]

    src_paths = f"batch_{i}/" + folder_paths + "/" + image_names
    no_batch_paths = folder_paths + "/" + image_names
    dst_paths = f"../processed/" + no_batch_paths

    df_paths = pd.DataFrame({
        "src_path": src_paths,
        "dst_path": dst_paths
    })

    print(len(df_paths))
    for j in range(len(df_paths)):
        src_path = Path(df_paths.iloc[j]["src_path"])
        dst_path = Path(df_paths.iloc[j]["dst_path"])

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_path, dst_path)
    print(f"Batch {i} files saved.")


