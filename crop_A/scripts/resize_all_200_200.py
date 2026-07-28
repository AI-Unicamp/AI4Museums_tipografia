import cv2
import os
from pathlib import Path

family_names = ['escritural', 'fantasia', 'grotesco', 'monograma', 'serifado', 'toscano']

for family in family_names:

    input_dir = Path(f"../data/interim/{family}")
    output_dir = input_dir / "resized"
    os.makedirs(output_dir, exist_ok=True)

    for file_path in input_dir.iterdir():
        if file_path.is_file():
            img = cv2.imread(str(file_path))
            if img is not None:
                resized_img = cv2.resize(img, (200, 200))
                output_path = output_dir / f"{file_path.stem}.png"
                print("Saved at ", output_path)
                cv2.imwrite(str(output_path), resized_img)