import cv2
import os
from pathlib import Path

family_names = ['escritural', 'fantasia', 'grotesco', 'monograma', 'serifado', 'toscano']
for family in family_names:

    input_dir = Path(f"../data/interim/{family}/resized")
    output_dir = Path(f"../data/interim/{family}/binarized")
    os.makedirs(output_dir, exist_ok=True)

    for file_path in input_dir.iterdir():
        if file_path.is_file():
            img = cv2.imread(str(file_path))
            if img is not None:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)
                output_path = output_dir / f"{file_path.stem}.png"
                print("Saved at ", output_path)
                cv2.imwrite(str(output_path), binary_img)