import shutil
from pathlib import Path

output_path = Path("/home/kkania/Datasets/scanning/nerfstudio/RF/00000000/images/")
input_path = Path("/home/kkania/Datasets/scanning/blender/rf-00-colmap/images/")

for image in input_path.glob("*.png"):
    index = int(image.stem.split("_")[0])
    new_name = "frame_{:05d}.png".format(index)
    shutil.copy(image, output_path / new_name)
    print(output_path / new_name)