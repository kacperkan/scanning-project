import enum
import gc
from collections import defaultdict
from typing import Dict, Iterator

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int32
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class ContractionType(enum.Enum):
    NONE = enum.auto()
    AABB = enum.auto()
    UN_BOUNDED_SPHERE = enum.auto()
    TANH = enum.auto()


def chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(
            *[
                (
                    arg[i : i + chunk_size]
                    if isinstance(arg, torch.Tensor)
                    else arg
                )
                for arg in args
            ],
            **kwargs,
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                "Return value of func must be in type [torch.Tensor, list,"
                f" tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)

    if out_type is None:
        return

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name.startswith("scale"):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0.0, scale_factor) / scale_factor
    elif name.startswith("clamp"):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0.0, clamp_max)
    elif name.startswith("mul"):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055
            - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name.startswith("+") or name.startswith("-"):
        return lambda x: x + float(name)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


class Constants:
    LATENT_DIM = 8


class Features(enum.Enum):
    ALPHA = enum.auto()
    BETA = enum.auto()
    SLOPE = enum.auto()
    DENSITY = enum.auto()
    NORMALS = enum.auto()
    RGB = enum.auto()
    VERTEX_ALPHA = enum.auto()  # from DMTet
    LATENT_FEATS = enum.auto()

    @property
    def dim(self) -> int:
        if self == Features.ALPHA:
            return 1
        if self == Features.BETA:
            return 1
        if self == Features.SLOPE:
            return 1
        if self == Features.DENSITY:
            return 1
        if self == Features.NORMALS:
            return 3
        if self == Features.RGB:
            return 3
        if self == Features.VERTEX_ALPHA:
            return 1
        if self == Features.LATENT_FEATS:
            return Constants.LATENT_DIM

    @staticmethod
    def from_str(a_str: str) -> "Features":
        if a_str == "alpha":
            return Features.ALPHA
        if a_str == "beta":
            return Features.BETA
        if a_str == "slope":
            return Features.SLOPE
        if a_str == "density":
            return Features.DENSITY
        if a_str == "normals":
            return Features.NORMALS
        if a_str == "rgb":
            return Features.RGB
        if a_str == "vertex_alpha":
            return Features.VERTEX_ALPHA
        if a_str == "latent_feats":
            return Features.LATENT_FEATS

    def get_num_features(self, rgb_basis_dim: int) -> int:
        if self == Features.RGB:
            return (rgb_basis_dim + 1) ** 2 * self.dim
        return self.dim

    def get_slice(
        self, feats_to_encode: list["Features"], rgb_basis_dim: int
    ) -> slice:
        start = 0
        for feat in feats_to_encode:
            if feat == self:
                return slice(
                    start, start + feat.get_num_features(rgb_basis_dim)
                )
            start += feat.get_num_features(rgb_basis_dim)
        raise ValueError(f"Feature {self} not found in {feats_to_encode}")

    @staticmethod
    def get_all_num_features(
        list_of_feats: list["Features"], rgb_basis_dim: int
    ) -> int:
        total = 0
        for feat in list_of_feats:
            total += feat.get_num_features(rgb_basis_dim)
        return total


def beta_activation(
    betas: Float[torch.Tensor, "... 1"],
) -> Float[torch.Tensor, "... 1"]:
    return nn.functional.softplus(betas - 2.3)


def extract_voxel_params(
    shape_feats: Float[torch.Tensor, "... shape_dim"],
    feats_to_encode: list[Features],
    rgb_basis_dim: int,
) -> Dict[Features, Float[torch.Tensor, "... dim"]]:

    output_dict = {}

    def add_to_dict_if_exists(feat: Features):
        if feat in feats_to_encode:
            output_dict[feat] = shape_feats[
                ..., feat.get_slice(feats_to_encode, rgb_basis_dim)
            ]

    for feat in feats_to_encode:
        add_to_dict_if_exists(feat)

    return output_dict


def bcc_mesh(
    res: int, chunk_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    p1s_odd = (
        torch.stack(
            torch.meshgrid(
                torch.arange(-1, res * 2, 2),
                torch.arange(-1, res * 2, 2),
                torch.arange(-1, res, 2),
            ),
            dim=-1,
        )
        .float()
        .view((-1, 3))
    )
    p1s_even = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res * 2, 2),
                torch.arange(0, res * 2, 2),
                torch.arange(0, res, 2),
            ),
            dim=-1,
        )
        .float()
        .view((-1, 3))
    )

    p1 = torch.cat([p1s_odd, p1s_even], dim=0)
    vec3 = lambda *val: torch.tensor(val, device=p1.device, dtype=p1.dtype)

    p1_all = p1
    offset = 0

    multipliers = torch.tensor([0.5, 0.5, 1.0])
    diff = torch.tensor([-1.0, -1.0, 0.0])

    def get_verts_edges_and_updated_offset(p, offset):
        tet_coords = torch.stack(p, dim=1)

        # volumes = (
        #     (
        #         torch.cross(
        #             tet_coords[:, 1] - tet_coords[:, 0],
        #             tet_coords[:, 2] - tet_coords[:, 0],
        #         )
        #         * (tet_coords[:, 3] - tet_coords[:, 0])
        #     )
        #     .sum(dim=-1)
        #     .abs()
        # )

        # keep_tets_mask = volumes == 4
        # tet_coords = tet_coords[keep_tets_mask]

        vertices = tet_coords.view(-1, 3)

        edges = []
        indices = torch.arange(len(tet_coords)) * 4  #  + offset
        # don't add offset because later in the code we process it sequentially

        edges1 = torch.stack([indices + 0, indices + 1, indices + 2], dim=-1)
        edges2 = torch.stack([indices + 1, indices + 2, indices + 3], dim=-1)
        edges3 = torch.stack([indices + 2, indices + 3, indices + 0], dim=-1)
        edges4 = torch.stack([indices + 3, indices + 0, indices + 1], dim=-1)

        edges = (
            torch.stack([edges1, edges2, edges3, edges4], dim=1).view((-1, 3))
            + offset
        )
        offset += len(tet_coords) * 4

        # mask = (tet_coords[..., [2]] % 2 == 1).float()
        # tet_coords = ((tet_coords + diff) * mask) + tet_coords * (1 - mask)

        return tet_coords, edges, offset

    for p1 in torch.split(p1_all, chunk_size):
        p2 = p1 + 1.0

        p3 = p1 + vec3(1.0, 1.0, -1.0)
        p4 = p1 + vec3(2.0, 0.0, 0.0)

        tet_coords, edges, offset = get_verts_edges_and_updated_offset(
            (p1, p2, p3, p4), offset
        )

        yield tet_coords, edges

        p3 = p1 + vec3(1.0, 1.0, -1.0)
        p4 = p1 + vec3(0.0, 2.0, 0.0)

        tet_coords, edges, offset = get_verts_edges_and_updated_offset(
            (p1, p2, p3, p4), offset
        )

        yield tet_coords, edges

        p3 = p1 + vec3(-1.0, 1.0, 1.0)
        p4 = p1 + vec3(0.0, 2.0, 0.0)
        tet_coords, edges, offset = get_verts_edges_and_updated_offset(
            (p1, p2, p3, p4), offset
        )

        yield tet_coords, edges

        p3 = p1 + vec3(1.0, -1.0, 1.0)
        p4 = p1 + vec3(2.0, 0.0, 0.0)
        tet_coords, edges, offset = get_verts_edges_and_updated_offset(
            (p1, p2, p3, p4), offset
        )

        yield tet_coords, edges

        p3 = p1 + vec3(1.0, -1.0, 1.0)
        p4 = p1 + vec3(0.0, 0.0, 2.0)
        tet_coords, edges, offset = get_verts_edges_and_updated_offset(
            (p1, p2, p3, p4), offset
        )

        yield tet_coords, edges

        p3 = p1 + vec3(-1.0, 1.0, 1.0)
        p4 = p1 + vec3(0.0, 0.0, 2.0)
        tet_coords, edges, offset = get_verts_edges_and_updated_offset(
            (p1, p2, p3, p4), offset
        )

        yield tet_coords, edges


def yet_another_meshing(
    res: int,
) -> tuple[Float[torch.Tensor, "n_verts 3"], Int32[torch.Tensor, "n_faces 3"]]:
    main_vertices = (
        np.stack(
            np.meshgrid(
                np.arange(0, res // 2),
                np.arange(0, res // 2),
                np.arange(0, res // 2),
                indexing="ij",
            ),
            axis=-1,
        )
        * 2
    )
    main_vertices = main_vertices.reshape((-1, 3)).tolist()

    edges = []
    triangles = []
    half_res = res // 2
    for i in range(half_res):
        for j in range(half_res):
            for k in range(half_res):
                offset = i * half_res**2 + j * half_res + k
                if k < half_res - 1:
                    edges.append((offset, offset + 1))
                if j < half_res - 1:
                    edges.append((offset, offset + half_res))
                if i < half_res - 1:
                    edges.append((offset, offset + half_res * half_res))

    new_id = half_res**3

    half_res_dec = half_res
    for i in range(0, half_res_dec):
        for j in range(0, half_res_dec):
            for k in range(0, half_res_dec):
                new_point = [i * 2 + 1, j * 2 + 1, k * 2 + 1]
                main_vertices.append(new_point)
                offset = i * half_res**2 + j * half_res + k

                edges.append((new_id, offset))
                if k < half_res - 1:
                    edges.append((new_id, offset + 1))
                    triangles.append((new_id, offset, offset + 1))
                if j < half_res - 1:
                    edges.append((new_id, offset + half_res))
                    triangles.append((new_id, offset, offset + half_res))
                if j < half_res - 1 and k < half_res - 1:
                    edges.append((new_id, offset + half_res + 1))
                    triangles.append(
                        (new_id, offset + half_res, offset + half_res + 1)
                    )
                    triangles.append(
                        (new_id, offset + 1, offset + half_res + 1)
                    )

                whole_row = half_res**2
                offset += half_res**2
                if i < half_res - 1:
                    edges.append((new_id, offset))
                    triangles.append((new_id, offset, offset - whole_row))
                    if k < half_res - 1:
                        edges.append((new_id, offset + 1))
                        triangles.append((new_id, offset, offset + 1))
                        triangles.append(
                            (new_id, offset + 1 - whole_row, offset + 1)
                        )
                    if j < half_res - 1:
                        edges.append((new_id, offset + half_res))
                        triangles.append((new_id, offset, offset + half_res))
                        triangles.append(
                            (
                                new_id,
                                offset + half_res - whole_row,
                                offset + half_res,
                            )
                        )
                    if j < half_res - 1 and k < half_res - 1:
                        edges.append((new_id, offset + half_res + 1))
                        triangles.append(
                            (new_id, offset + half_res, offset + half_res + 1)
                        )
                        triangles.append(
                            (new_id, offset + 1, offset + half_res + 1)
                        )

                        triangles.append(
                            (
                                new_id,
                                offset + half_res + 1 - whole_row,
                                offset + half_res + 1,
                            )
                        )

                offset -= whole_row
                if k < half_res_dec - 1:
                    edges.append((new_id, new_id + 1))
                    triangles.append((new_id, new_id + 1, offset + 1))
                if j < half_res_dec - 1:
                    edges.append((new_id, new_id + half_res_dec))
                    triangles.append(
                        (new_id, new_id + half_res_dec, offset + half_res)
                    )
                if j < half_res - 1 and k < half_res - 1:
                    triangles.append(
                        (new_id, new_id + 1, offset + half_res + 1)
                    )
                    triangles.append(
                        (new_id, new_id + half_res_dec, offset + half_res + 1)
                    )
                if i < half_res_dec - 1:
                    lower_id = new_id + half_res_dec * half_res_dec
                    edges.append((new_id, lower_id))

                offset += whole_row
                if i < half_res_dec - 1:
                    triangles.append((new_id, lower_id, offset))
                    if k < half_res_dec - 1:
                        triangles.append((new_id, new_id + 1, offset + 1))
                        triangles.append((new_id, lower_id, offset + 1))
                    if j < half_res_dec - 1:
                        triangles.append(
                            (new_id, new_id + half_res_dec, offset + half_res)
                        )
                        triangles.append((new_id, lower_id, offset + half_res))
                    if j < half_res - 1 and k < half_res - 1:
                        triangles.append(
                            (new_id, new_id + 1, offset + half_res + 1)
                        )
                        triangles.append(
                            (
                                new_id,
                                new_id + half_res_dec,
                                offset + half_res + 1,
                            )
                        )
                        triangles.append(
                            (new_id, lower_id, offset + half_res + 1)
                        )

                new_id += 1
    vertices_array = np.array(main_vertices, dtype=np.float32)

    vertices_tensor = torch.from_numpy(vertices_array)

    triangles_array = np.array(triangles, dtype=np.int32)
    triangles_tensor = torch.from_numpy(triangles_array).long()
    return vertices_tensor, triangles_tensor
