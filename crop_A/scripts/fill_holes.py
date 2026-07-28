import cv2
import numpy as np
import os
from pathlib import Path

family_names = ['escritural', 'fantasia', 'grotesco', 'monograma', 'serifado', 'toscano']
kernel = np.ones((6,6), np.uint8)

for family in family_names:

    input_dir = Path(f"../data/interim/{family}/binarized")
    output_dir = Path(f"../data/interim/{family}/filled")
    os.makedirs(output_dir, exist_ok=True)

    for file_path in input_dir.iterdir():
        if file_path.is_file():
            img = cv2.imread(str(file_path))
            if img is not None:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = 255-gray_img
                closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
                closing = 255-closing
                output_path = output_dir / f"{file_path.stem}.png"
                print("Saved at ", output_path)
                cv2.imwrite(str(output_path), closing)