import argparse
import copy
import json
import shutil
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm


def normalize_poses(poses):
    # estimation scene center as the average of the intersection of s
    # elected pairs of camera rays
    cams_ori = poses[..., 3]
    cams_dir = poses[:, :3, :3] @ torch.as_tensor([0.0, 0.0, -1.0])
    cams_dir = F.normalize(cams_dir, dim=-1)
    A = torch.stack([cams_dir, -cams_dir.roll(1, 0)], dim=-1)
    b = -cams_ori + cams_ori.roll(1, 0)

    t = torch.linalg.lstsq(A, b).solution
    center = (
        torch.stack([cams_dir, cams_dir.roll(1, 0)], dim=-1) * t[:, None, :]
        + torch.stack([cams_ori, cams_ori.roll(1, 0)], dim=-1)
    ).mean((0, 2))

    z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    y_ = torch.as_tensor([z[1], -z[0], 0.0])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)
    # rotation and translation
    Rc = torch.stack([x, y, z], dim=1)
    tc = center.reshape(3, 1)
    R, t = Rc.T, -Rc.T @ tc
    poses_homo = torch.cat(
        [
            poses,
            torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(poses.shape[0], -1, -1),
        ],
        dim=1,
    )
    inv_trans = torch.cat(
        [torch.cat([R, t], dim=1), torch.as_tensor([[0.0, 0.0, 0.0, 1.0]])],
        dim=0,
    )
    poses_norm = (inv_trans @ poses_homo)[:, :3]  # (N_images, 4, 4)

    # scaling
    scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
    poses_norm[..., 3] /= scale

    # apply the transformation to the point cloud

    return poses_norm


parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str)
parser.add_argument(
    "--masks_folder", type=str, required=False, help="Path to the masks folder if available", default=None
)
parser.add_argument("output_dir", type=str)
parser.add_argument("--downscale", type=float, default=1.0, help="Uniform scaling for images")
parser.add_argument("--reverse_right", action="store_true")
parser.add_argument("--indices_to_ignore", nargs="*", type=int)


args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
masks_folder = Path(args.masks_folder)

if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

nerfstudio_folder_dir = input_dir
with open(nerfstudio_folder_dir / "transforms.json") as f:
    nerfstudio_cams = json.loads(f.read())

has_foreground_mask = args.masks_folder is not None
if has_foreground_mask:
    try:
        indices_to_masks = {int(path.with_suffix("").name.split("_")[0]): path for path in masks_folder.glob("*.png")}
        frame_first = False
    except ValueError:
        frame_first = True
        indices_to_masks = {int(path.with_suffix("").name.split("_")[1]): path for path in masks_folder.glob("*.png")}
    if frame_first:
        print("Format for masks was frame_<num>.png")
    else:
        print("Format for masks was <num>_rgb.png")
else:
    indices_to_masks = {}


output_transforms = copy.deepcopy(nerfstudio_cams)
output_transforms["frames"] = []

translations = []
poses = []
for frame in nerfstudio_cams["frames"]:
    c2w = np.array(frame["transform_matrix"]).reshape((4, 4))
    c2w[:, 1] *= -1
    c2w[1] *= -1
    c2w[:, 2] *= -1

    w = int(frame["w"] / args.downscale)
    h = int(frame["h"] / args.downscale)
    fl_x = frame["fl_x"] / args.downscale
    fl_y = frame["fl_y"] / args.downscale
    cx = frame["cx"] / args.downscale
    cy = frame["cy"] / args.downscale

    rgb_path = frame["file_path"]
    camtoworld = c2w.tolist()
    intrisics = np.array(
        [
            [fl_x, 0, cx, 0.0],
            [0, fl_y, cy, 0.0],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1.0],
        ]
    ).tolist()
    output_transforms["height"] = h
    output_transforms["width"] = w

    image_path_normal = rgb_path.replace("\\", "/")  # windows fix
    frame["rgb_path"] = str(image_path_normal)
    new_frame = {
        "rgb_path": image_path_normal,
        "camtoworld": camtoworld,
        "intrinsics": intrisics,
        "k1": frame.get("k1", 0.0),
        "k2": frame.get("k2", 0.0),
        "k3": frame.get("k3", 0.0),
        "p1": frame.get("p1", 0.0),
        "p2": frame.get("p2", 0.0),
    }

    # if int(Path(image_path_normal).with_suffix("").name.split("_")[-1]) in args.indices_to_ignore:
    #     print("Ignoring ...", image_path_normal)
    #     continue
    output_transforms["frames"].append(new_frame)
    translations.append(c2w[:3, 3])
    poses.append(c2w)

poses = torch.tensor(poses).float()
poses = normalize_poses(poses[:, :3])

for i, pose in enumerate(poses):
    c2w = pose.numpy()
    output_transforms["frames"][i]["camtoworld"] = c2w.tolist()


max_span = np.max(np.abs(np.array(translations))) * 2

aabb = np.array([[-max_span, -max_span, -max_span], [max_span, max_span, max_span]]).tolist()

output_transforms["scene_box"] = {
    "aabb": aabb,
    "near": 0.01 * np.linalg.norm(max_span),
    "far": 100.0 * np.linalg.norm(max_span),
    "radius": 2.0 * np.linalg.norm(max_span),
    "collider_type": "near_far",
}


shutil.copytree(nerfstudio_folder_dir / "images", output_dir / "images")
if has_foreground_mask:
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

kernel = np.ones((11, 11), np.uint8)

for frame in tqdm.tqdm(output_transforms["frames"]):
    image_path = Path(frame["rgb_path"])
    img = None
    if args.downscale != 1.0:
        if img is None:
            img = imageio.imread(output_dir / image_path)
        img = cv2.resize(img, (int(img.shape[1] / args.downscale), int(img.shape[0] / args.downscale)))
        imageio.imwrite(output_dir / image_path, img)

    if has_foreground_mask:

        try:
            index = int(image_path.with_suffix("").name.split("_")[1])
        except ValueError:
            index = int(image_path.with_suffix("").name.split("_")[0])
        mask_path = indices_to_masks[index]
        assert mask_path.exists()

        if img is None:
            img = imageio.imread(output_dir / image_path)
        mask = imageio.imread(mask_path)
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if mask.ndim == 2:
            mask = np.repeat(mask[:, :, None], 3, axis=2)

        imageio.imwrite(output_dir / "masks" / image_path.name, mask)
        frame["foreground_mask"] = str(Path("masks") / image_path.name)

output_transforms["has_foreground_mask"] = has_foreground_mask
with open(output_dir / "meta_data.json", "w") as f:
    json.dump(output_transforms, f, indent=4)
