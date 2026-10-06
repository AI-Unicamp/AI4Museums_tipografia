import numpy as np
import pandas as pd
import os


for i in range(1,4):
    path = f"annotations_batch_{i}.csv"
    print(f"Current processing file: {path}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    unique_names = df["image_name"].unique()
    selected = []
    for name in unique_names:
        unique_image = df[df["image_name"] == name]
        count = np.sum(unique_image["is_good"])
        if count >= 2:
            if 'folder_path' in unique_image.columns:
                folder = unique_image['folder_path'].iloc[0]
            else:
                folder = ''
            selected.append({'folder_path': folder, 'image_name': name})

    out_df = pd.DataFrame(selected, columns=['folder_path', 'image_name'])
    out_df.to_csv(f"selected_batch_{i}.csv", index=False)
    print(f"Wrote {len(out_df)} rows to selected_batch_{i}.csv\n")

