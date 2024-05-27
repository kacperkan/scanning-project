import math
from typing import Optional

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
from jaxtyping import Float
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_debug,
    rank_zero_info,
)

from models.utils import Features, bcc_mesh, get_activation
from systems.utils import update_module_step
from utils.misc import config_to_primitive, get_rank


class VanillaFrequency(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.N_freqs = config["n_frequencies"]
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(
            0, self.N_freqs - 1, self.N_freqs
        )
        self.n_output_dims = self.in_channels * (
            len(self.funcs) * self.N_freqs
        )
        self.n_masking_step = config.get("n_masking_step", 0)
        self.update_step(
            None, None
        )  # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq * x) * mask]
        return torch.cat(out, -1)

    def update_step(self, epoch, global_step):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            self.mask = (
                1.0
                - torch.cos(
                    math.pi
                    * (
                        global_step / self.n_masking_step * self.N_freqs
                        - torch.arange(0, self.N_freqs)
                    ).clamp(0, 1)
                )
            ) / 2.0
            rank_zero_debug(
                f"Update mask: {global_step}/{self.n_masking_step} {self.mask}"
            )


class ProgressiveBandHashGrid(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config["otype"] = "HashGrid"
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config["n_levels"]
        self.n_features_per_level = config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            config["start_level"],
            config["start_step"],
            config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, epoch, global_step):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            rank_zero_info(f"Update grid level to {current_level}")
        self.current_level = current_level
        self.mask[: self.current_level * self.n_features_per_level] = 1.0


class BCCEncoding(nn.Module):
    def __init__(self, n_input_dims, config) -> None:
        super().__init__()

        self.n_input_dims = n_input_dims
        self.config = config

        self.grid_resolution = config["grid_resolution"]
        self.point_dims = n_input_dims
        self.feats_to_encode = [
            Features.from_str(feat) for feat in config["feats_to_encode"]
        ]
        if config["stored_data"] == "latent":
            self.latent_dim = config["feats_per_level_if_latent"]
        else:
            self.latent_dim = Features.get_all_num_features(
                self.feats_to_encode, config["sh_n_bases"]
            )
        self.basis_dim = config["sh_n_bases"]

        self._x_dim = self.grid_resolution
        self._y_dim = self.grid_resolution
        self._z_dim = self.grid_resolution

        self.register_buffer(
            "scene_radius", torch.tensor(config["scene_radius"])
        )
        self.register_buffer(
            "primes",
            torch.tensor([1, 2654435761, 805459861], dtype=torch.long),
        )

        self.clamp_x = lambda x: x.clamp(0, self.x_dim - 1)
        self.clamp_y = lambda x: x.clamp(0, self.y_dim - 1)
        self.clamp_z = lambda x: x.clamp(0, self.z_dim - 1)
        self.register_buffer("index_conversion", torch.tensor([0.5, 0.5, 1.0]))

        if config["stored_data"] == "latent":
            self.initialize_latents()
        else:
            self.initialize_direct_storage()

    def get_total_samples(self) -> int:
        return self.get_total_elements_in_grid()

    def get_total_elements_in_grid(self) -> int:
        return int(self.grid_resolution**self.point_dims)

    def initialize_direct_storage(self):

        self.latents = nn.Embedding(self.get_total_samples(), self.latent_dim)

        # offset = (1 / self.grid_resolution) / 2  # centers
        offset = 0
        a_min = -1 + offset * 2  # 2 because range is [-1, 1]
        a_max = 1 - offset * 2  # same here
        xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(a_min, a_max, self.x_dim),
                torch.linspace(a_min, a_max, self.y_dim),
                torch.linspace(a_min, a_max, self.z_dim),
                indexing="ij",
            ),
            dim=-1,
        ).view((-1, self.point_dims))

        self.register_buffer(
            "bases",
            torch.tensor(
                [self.grid_resolution**2, self.grid_resolution, 1],
                dtype=torch.long,
            ),
        )

        self.register_buffer("grid_coords", xyz)

        feats_list = self.feats_to_encode
        if Features.ALPHA in feats_list:
            self.latents.weight.data[
                ..., Features.ALPHA.get_slice(feats_list, self.basis_dim)
            ] = -math.pi  # alpha
        if Features.BETA in feats_list:
            self.latents.weight.data[
                ..., Features.BETA.get_slice(feats_list, self.basis_dim)
            ] = -1  # beta
        if (feat := Features.SLOPE) in feats_list:
            self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ] = -0.0001  # slopes
        if (feat := Features.VERTEX_ALPHA) in feats_list:
            data_slice = self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ]
            self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ] = (
                torch.randn_like(data_slice) * 0.01
            )  # vertex alpha from DMTet
        if (feat := Features.DENSITY) in feats_list:
            transformed_xyz = self.init_coordinate_transform(xyz)

            # self.latents.weight.data[
            #     ..., feat.get_slice(feats_list, self.basis_dim)
            # ] = (
            #     transformed_xyz.norm(dim=-1, keepdim=True)
            #     - self.config["sphere_init_radius"]
            # )  # sdf
            verts = torch.cat(
                [v for v, _ in bcc_mesh(self.grid_resolution, 16384)], dim=0
            )
            verts = (verts * torch.tensor([0.5, 0.5, 1.0])).floor()
            verts = verts.view((-1, 3))
            x, y, z = verts.unbind(-1)
            indices = (
                x * self.grid_resolution**2 + y * self.grid_resolution + z
            )
            indices = indices.clamp(0, self.get_total_samples() - 1).long()

            def unique(x, dim=0):
                unique, inverse, counts = torch.unique(
                    x,
                    dim=dim,
                    sorted=True,
                    return_inverse=True,
                    return_counts=True,
                )
                inv_sorted = inverse.argsort(stable=True)
                tot_counts = torch.cat(
                    (counts.new_zeros(1), counts.cumsum(dim=0))
                )[:-1]
                index = inv_sorted[tot_counts]
                return unique, inverse, counts, index

            indices, _, _, unique_indices = unique(indices)
            verts = verts[unique_indices] / (self.grid_resolution - 1)
            verts = verts * 2 - 1
            self.latents.weight.data[
                indices, feat.get_slice(feats_list, self.basis_dim)
            ] = (
                verts.norm(dim=-1, keepdim=True)
                - self.config["sphere_init_radius"]
            )  # sdf
        if (feat := Features.RGB) in feats_list:
            data = (
                torch.rand_like(
                    self.latents.weight.data[
                        ..., feat.get_slice(feats_list, self.basis_dim)
                    ]
                )
                * 2
                - 1
            ) * 1e-2

            self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ] = data  # rgb data

        if (feat := Features.NORMALS) in feats_list:
            self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ] = nn.functional.normalize(
                xyz, dim=-1
            )  # normals

        if (feat := Features.LATENT_FEATS) in feats_list:
            a_slice = feat.get_slice(feats_list, self.basis_dim)
            self.latents.weight.data[..., a_slice] = (
                torch.rand_like(self.latents.weight.data[..., a_slice]) * 2e-5
                - 1e-5
            )

        self._grid = None

    @property
    def n_output_dims(self) -> int:
        return self.latent_dim

    @property
    def x_dim(self) -> int:
        return self.grid_resolution

    @property
    def y_dim(self) -> int:
        return self.grid_resolution

    @property
    def z_dim(self) -> int:
        return self.grid_resolution

    def forward(self, x):
        latents = self.get_latents(x)
        return latents

    def get_latents(
        self,
        pts: Float[torch.Tensor, "batch dim"],
        pre_interpolation: Optional[bool] = None,
    ):
        if pre_interpolation is not None:
            flag = pre_interpolation
        else:
            flag = self.config["pre_interpolation"]
        return self._interpolate_linear(pts, flag)

    def _interpolate_linear(
        self,
        pts: Float[torch.Tensor, "batch pts_dim"],
        pre_interpolation: bool = True,
    ) -> Float[torch.Tensor, "batch n_input_dims"]:

        tet_coords, _ = self.get_tet_coords_and_barycentrics_v2(pts)
        p1, p2, p3, p4 = tet_coords.unbind(-2)

        d1 = self.get_data_for_point(p1, None)
        d2 = self.get_data_for_point(p2, None)
        d3 = self.get_data_for_point(p3, None)
        d4 = self.get_data_for_point(p4, None)

        vals = torch.stack([d1, d2, d3, d4], dim=-2)
        if pre_interpolation:
            return self.interpolate_data(pts, vals)[0]
        return vals

    def interpolate_data(
        self,
        pts: Float[torch.Tensor, "pts pts_dim"],
        data: Float[torch.Tensor, "pts entries data_dim"],
        interpolators: Optional[Float[torch.Tensor, "pts 4"]] = None,
    ) -> Float[torch.Tensor, "pts data_dim"]:
        if interpolators is None:
            interpolators = self.get_barycentrics(pts)
        d1, d2, d3, d4 = data.unbind(-2)
        vals = torch.stack([d1, d3 - d1, d2 - d4, d4 - d3], dim=-2)
        # vals = torch.stack([d4, d4 - d4, d4 - d4, d4 - d4], dim=-2)
        data = torch.einsum("...ji,...j->...i", vals, interpolators)
        return data, interpolators

    def get_data_for_point(
        self,
        point: Float[torch.Tensor, "pts dim"],
        grid: Optional[Float[torch.Tensor, "num_latents dim"]] = None,
    ) -> Float[torch.Tensor, "pts feat_dim"]:
        # TODO
        point = (point * self.index_conversion).floor()
        # point = point.floor()
        grid = self.latents.weight

        p_x, p_y, p_z = point.long().unbind(-1)

        new_coords = (
            self.clamp_x(p_x) * self.y_dim * self.z_dim
            + self.clamp_y(p_y) * self.z_dim
            + self.clamp_z(p_z)
        )
        data = grid[new_coords]

        return data

    def get_barycentrics(
        self, pts: Float[torch.Tensor, "pts dim"]
    ) -> Float[torch.Tensor, "pts 3"]:
        # pts = pts.sort(dim=-1, descending=True).values
        # pts = pts.clamp(0, 1)
        bcc_pts = pts
        n_pts = bcc_pts.shape[0]
        size_volume = torch.tensor(
            [
                2 * self.grid_resolution - 1,
                2 * self.grid_resolution - 1,
                self.grid_resolution - 1,
            ],
            device=bcc_pts.device,
        )

        bcc_pts = bcc_pts * size_volume
        x, y, z = bcc_pts.unbind(-1)
        abc = torch.stack([x + y, x + z, y + z], dim=-1) * 0.5
        floors = abc.floor()

        abc = abc - floors

        sorting = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=x.device, dtype=x.dtype
        )

        sorting = sorting[None, ...].expand(n_pts, -1).clone()
        sorting[:, 1] = abc.max(dim=-1).values
        sorting[:, 2] = abc.min(dim=-1).values
        sorting[:, 3] = abc.sum(dim=-1) - sorting[:, 1] - sorting[:, 2]

        return sorting

    def get_tet_coords_and_barycentrics(
        self, pts: Float[torch.Tensor, "pts dim"]
    ) -> tuple[
        Float[torch.Tensor, "pts 4 dim"],
        Float[torch.Tensor, "pts 3"],
    ]:
        # pts = pts.sort(dim=-1, descending=True).values
        # pts = pts.clamp(0, 1)
        bcc_pts = pts
        n_pts = bcc_pts.shape[0]
        size_volume = torch.tensor(
            [
                2 * self.grid_resolution - 1,
                2 * self.grid_resolution - 1,
                self.grid_resolution - 1,
            ],
            device=bcc_pts.device,
        )

        bcc_pts = bcc_pts * size_volume
        x, y, z = bcc_pts.unbind(-1)
        abc = torch.stack([x + y, x + z, y + z], dim=-1) * 0.5
        floors = abc.floor()

        abc = abc - floors

        floors_x, floors_y, floors_z = floors.unbind(-1)
        p1 = torch.stack(
            [
                floors_x + floors_y - floors_z,
                floors_x - floors_y + floors_z,
                -floors_x + floors_y + floors_z,
            ],
            dim=-1,
        )

        sorting = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=x.device, dtype=x.dtype
        )
        sorting = sorting[None, ...].expand(n_pts, -1).clone()
        sorting[:, 1] = abc.max(dim=-1).values
        sorting[:, 2] = abc.min(dim=-1).values
        sorting[:, 3] = abc.sum(dim=-1) - sorting[:, 1] - sorting[:, 2]
        import ipdb

        ipdb.set_trace()

        vec3 = lambda val: torch.tensor(val, device=x.device, dtype=x.dtype)
        multipliers = vec3([0.5, 0.5, 1.0])
        diff = vec3([-1.0, -1.0, 0.0])

        # def pt_to_index(pt):
        #     mask = (pt[..., [2]] % 2 == 1).float()
        #     pt = ((pt + diff) * mask) + pt * (1 - mask)
        #     return pt  # * multipliers

        # p1 = ((p1 + diff) * mask) + p1 * (1 - mask)

        p2 = p1 + vec3([1.0, 1.0, 1.0])

        a, b, c = abc.unbind(-1)
        a = a.unsqueeze(dim=-1)
        b = b.unsqueeze(dim=-1)
        c = c.unsqueeze(dim=-1)
        a_b_c = ((a >= b) & (b >= c)).float()
        a_c_b = ((a >= b) & (a >= c)).float()
        c_a_b = (a >= b).float() - a_b_c - a_c_b

        b_a_c = ((a < b) & (a >= c)).float()
        b_c_a = ((a < b) & (b >= c)).float()
        c_b_a = (a < b).float() - b_a_c - b_c_a

        p3_a_b_c = vec3([1.0, 1.0, -1.0]) * a_b_c
        p4_a_b_c = vec3([2.0, 0.0, 0.0]) * a_b_c

        p3_a_c_b = vec3([1.0, 1.0, -1.0]) * a_c_b
        p4_a_c_b = vec3([0.0, 2.0, 0.0]) * a_c_b

        p3_c_a_b = vec3([-1.0, 1.0, 1.0]) * c_a_b
        p4_c_a_b = vec3([0.0, 2.0, 0.0]) * c_a_b

        p3_b_a_c = vec3([1.0, -1.0, 1.0]) * b_a_c
        p4_b_a_c = vec3([2.0, 0.0, 0.0]) * b_a_c

        p3_b_c_a = vec3([1.0, -1.0, 1.0]) * b_c_a
        p4_b_c_a = vec3([0.0, 0.0, 2.0]) * b_c_a

        p3_c_b_a = vec3([-1.0, 1.0, 1.0]) * c_b_a
        p4_c_b_a = vec3([0.0, 0.0, 2.0]) * c_b_a

        p3 = (
            p1
            + p3_a_b_c
            + p3_a_c_b
            + p3_c_a_b
            + p3_b_a_c
            + p3_b_c_a
            + p3_c_b_a
        )
        p4 = (
            p1
            + p4_a_b_c
            + p4_a_c_b
            + p4_c_a_b
            + p4_b_a_c
            + p4_b_c_a
            + p4_c_b_a
        )

        return (torch.stack([p1, p2, p3, p4], dim=-2), sorting)

    def get_tet_coords_and_barycentrics_v2(
        self, pts: Float[torch.Tensor, "pts dim"]
    ) -> tuple[
        Float[torch.Tensor, "pts 4 dim"],
        Float[torch.Tensor, "pts 3"],
    ]:
        # pts = pts.sort(dim=-1, descending=True).values
        # pts = pts.clamp(0, 1)
        bcc_pts = pts
        n_pts = bcc_pts.shape[0]
        size_volume = torch.tensor(
            [
                2 * self.grid_resolution - 1,
                2 * self.grid_resolution - 1,
                self.grid_resolution - 1,
            ],
            device=bcc_pts.device,
        )

        bcc_pts = bcc_pts * size_volume
        x, y, z = bcc_pts.unbind(-1)
        abc = torch.stack([x + y, x + z, y + z], dim=-1) * 0.5
        floors = abc.floor()

        abc = abc - floors

        floors_x, floors_y, floors_z = floors.unbind(-1)
        p1 = torch.stack(
            [
                floors_x + floors_y - floors_z,
                floors_x - floors_y + floors_z,
                -floors_x + floors_y + floors_z,
            ],
            dim=-1,
        )
        p2 = p1 + torch.tensor([1.0, 1.0, 1.0], device=x.device, dtype=x.dtype)
        p3 = p1 + torch.tensor(
            [1.0, 1.0, -1.0], device=x.device, dtype=x.dtype
        )
        p4 = p1 + torch.tensor([2.0, 0.0, 0.0], device=x.device, dtype=x.dtype)

        sorting = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=x.device, dtype=x.dtype
        )

        sorting = sorting[None, ...].expand(n_pts, -1).clone()
        sorting[:, 1] = abc.max(dim=-1).values
        sorting[:, 2] = abc.min(dim=-1).values
        sorting[:, 3] = abc.sum(dim=-1) - sorting[:, 1] - sorting[:, 2]

        vec3 = lambda *val: torch.tensor(val, device=x.device, dtype=x.dtype)

        def are_close(a, b, eps=1e-18):
            return ((a - b).abs() < eps).float()

        p3 = (
            p3
            + are_close(sorting[:, [1]], abc[:, [1]]) * vec3(0.0, -2.0, 2.0)
            + are_close(sorting[:, [1]], abc[:, [2]]) * vec3(-2.0, 0.0, 2.0)
        )

        p4 = (
            p4
            + are_close(sorting[:, [2]], abc[:, [0]]) * vec3(-2.0, 0.0, 2.0)
            + are_close(sorting[:, [2]], abc[:, [1]]) * vec3(-2.0, 2.0, 0.0)
        )

        return torch.stack([p1, p2, p3, p4], dim=-2), sorting

    def init_coordinate_transform(self, coords: Float[torch.Tensor, "pts 3"]):
        coords = coords * 0.5 + 0.5
        multipliers = torch.tensor([2.0, 2.0, 1.0])
        coords = coords * (self.grid_resolution - 1)
        coords = coords * multipliers
        x, y, z = coords.unbind(-1)
        mask = ((z.long() % 2) == 1).float()
        x = x * (1 - mask) + (x + 1) * mask
        y = y * (1 - mask) + (y + 1) * mask
        coords = torch.stack([x, y, z], dim=-1) / (self.grid_resolution - 1)
        coords = coords / multipliers
        coords = coords * 2 - 1

        # pts, barys = self.get_tet_coords_and_barycentrics(coords)
        # data = pts * self.index_conversion
        # data = data / (self.grid_resolution - 1)
        # data = data * 2 - 1
        # middle = self.interpolate_data(pts, data, barys)[0]

        return coords


class CCEncoding(nn.Module):
    def __init__(self, n_input_dims, config) -> None:
        super().__init__()

        self.n_input_dims = n_input_dims
        self.config = config

        self.grid_resolution = config["grid_resolution"]
        self.point_dims = n_input_dims
        self.feats_to_encode = [
            Features.from_str(feat) for feat in config["feats_to_encode"]
        ]
        if config["stored_data"] == "latent":
            self.latent_dim = config["feats_per_level_if_latent"]
        else:
            self.latent_dim = Features.get_all_num_features(
                self.feats_to_encode, config["sh_n_bases"]
            )
        self.basis_dim = config["sh_n_bases"]

        self._x_dim = self.grid_resolution
        self._y_dim = self.grid_resolution
        self._z_dim = self.grid_resolution

        self.register_buffer(
            "scene_radius", torch.tensor(config["scene_radius"])
        )
        self.register_buffer(
            "primes",
            torch.tensor([1, 2654435761, 805459861], dtype=torch.long),
        )

        self.clamp_x = lambda x: x.clamp(0, self.x_dim - 1)
        self.clamp_y = lambda x: x.clamp(0, self.y_dim - 1)
        self.clamp_z = lambda x: x.clamp(0, self.z_dim - 1)
        self.register_buffer("index_conversion", torch.tensor([0.5, 0.5, 1.0]))

        if config["stored_data"] == "latent":
            self.initialize_latents()
        else:
            self.initialize_direct_storage()

    def get_total_samples(self) -> int:
        return self.get_total_elements_in_grid()

    def get_total_elements_in_grid(self) -> int:
        return int(self.grid_resolution**self.point_dims)

    def initialize_direct_storage(self):

        self.latents = nn.Embedding(self.get_total_samples(), self.latent_dim)

        # offset = (1 / self.grid_resolution) / 2  # centers
        offset = 0
        a_min = -1 + offset * 2  # 2 because range is [-1, 1]
        a_max = 1 - offset * 2  # same here
        xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(a_min, a_max, self.x_dim),
                torch.linspace(a_min, a_max, self.y_dim),
                torch.linspace(a_min, a_max, self.z_dim),
                indexing="ij",
            ),
            dim=-1,
        ).view((-1, self.point_dims))

        self.register_buffer(
            "bases",
            torch.tensor(
                [self.grid_resolution**2, self.grid_resolution, 1],
                dtype=torch.long,
            ),
        )

        self.register_buffer("grid_coords", xyz)

        feats_list = self.feats_to_encode
        if Features.ALPHA in feats_list:
            self.latents.weight.data[
                ..., Features.ALPHA.get_slice(feats_list, self.basis_dim)
            ] = -math.pi  # alpha
        if Features.BETA in feats_list:
            self.latents.weight.data[
                ..., Features.BETA.get_slice(feats_list, self.basis_dim)
            ] = -1  # beta
        if (feat := Features.SLOPE) in feats_list:
            self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ] = -0.0001  # slopes
        if (feat := Features.VERTEX_ALPHA) in feats_list:
            data_slice = self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ]
            self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ] = (
                torch.randn_like(data_slice) * 0.01
            )  # vertex alpha from DMTet
        if (feat := Features.DENSITY) in feats_list:
            transformed_xyz = self.init_coordinate_transform(xyz)

            self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ] = (
                transformed_xyz.norm(dim=-1, keepdim=True)
                - self.config["sphere_init_radius"]
            )  # sdf
        if (feat := Features.RGB) in feats_list:
            data = (
                torch.rand_like(
                    self.latents.weight.data[
                        ..., feat.get_slice(feats_list, self.basis_dim)
                    ]
                )
                * 2
                - 1
            ) * 1e-2

            self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ] = data  # rgb data

        if (feat := Features.NORMALS) in feats_list:
            self.latents.weight.data[
                ..., feat.get_slice(feats_list, self.basis_dim)
            ] = nn.functional.normalize(
                xyz, dim=-1
            )  # normals

        if (feat := Features.LATENT_FEATS) in feats_list:
            a_slice = feat.get_slice(feats_list, self.basis_dim)
            self.latents.weight.data[..., a_slice] = (
                torch.rand_like(self.latents.weight.data[..., a_slice]) * 2e-5
                - 1e-5
            )

        self._grid = None

    @property
    def n_output_dims(self) -> int:
        return self.latent_dim

    @property
    def x_dim(self) -> int:
        return self.grid_resolution

    @property
    def y_dim(self) -> int:
        return self.grid_resolution

    @property
    def z_dim(self) -> int:
        return self.grid_resolution

    def forward(self, x):
        latents = self.get_latents(x)
        return latents

    def get_latents(
        self,
        pts: Float[torch.Tensor, "batch dim"],
        pre_interpolation: Optional[bool] = None,
    ):
        if pre_interpolation is not None:
            flag = pre_interpolation
        else:
            flag = self.config["pre_interpolation"]
        return self._interpolate_linear(pts, flag)

    def _interpolate_nn(
        self,
        pts: Float[torch.Tensor, "batch pts z_samples dim"],
        pre_interpolation: bool = True,
    ) -> Float[torch.Tensor, "batch pts z_samples feat_dim"]:
        pts = pts * (self.grid_resolution - 2)

        l = pts.round()

        lx, ly, lz = l.unbind(-1)
        feats = self.get_data_for_point([lx, ly, lz], None)
        if not pre_interpolation:
            return feats.unsqueeze(dim=-1)

        return feats

    def _interpolate_linear(
        self,
        pts: Float[torch.Tensor, "batch pts z_samples dim"],
        pre_interpolation: bool = True,
    ) -> Float[torch.Tensor, "batch pts z_samples feat_dim"]:
        original_pts = pts
        pts = pts * (self.grid_resolution - 2)
        l = pts.floor()

        lx, ly, lz = l.unbind(-1)
        data000 = self.get_data_for_point([lx, ly, lz], None)
        data001 = self.get_data_for_point([lx, ly, lz + 1], None)
        data010 = self.get_data_for_point([lx, ly + 1, lz], None)
        data011 = self.get_data_for_point([lx, ly + 1, lz + 1], None)
        data100 = self.get_data_for_point([lx + 1, ly, lz], None)
        data101 = self.get_data_for_point([lx + 1, ly, lz + 1], None)
        data110 = self.get_data_for_point([lx + 1, ly + 1, lz], None)
        data111 = self.get_data_for_point([lx + 1, ly + 1, lz + 1], None)
        data = torch.stack(
            [
                data000,
                data001,
                data010,
                data011,
                data100,
                data101,
                data110,
                data111,
            ],
            dim=1,
        )

        if pre_interpolation:
            return self.interpolate_data(original_pts, data)[0]
        return data

    def interpolate_data(
        self,
        pts: Float[torch.Tensor, "pts pts_dim"],
        data: Float[torch.Tensor, "pts entries data_dim"],
        interpolators: Optional[Float[torch.Tensor, "pts ..."]] = None,
    ) -> Float[torch.Tensor, "pts data_dim"]:
        if interpolators is None:
            interpolators = self.get_interpolators(pts)
        wa, wb = interpolators.unbind(-1)
        (
            data000,
            data001,
            data010,
            data011,
            data100,
            data101,
            data110,
            data111,
        ) = data.unbind(1)

        c00 = data000 * wa[..., 2:] + data001 * wb[..., 2:]
        c01 = data010 * wa[..., 2:] + data011 * wb[..., 2:]
        c10 = data100 * wa[..., 2:] + data101 * wb[..., 2:]
        c11 = data110 * wa[..., 2:] + data111 * wb[..., 2:]
        c0 = c00 * wa[..., 1:2] + c01 * wb[..., 1:2]
        c1 = c10 * wa[..., 1:2] + c11 * wb[..., 1:2]
        data = c0 * wa[..., :1] + c1 * wb[..., :1]
        return data, interpolators

    def get_interpolators(
        self, pts: Float[torch.Tensor, "pts dim"]
    ) -> Float[torch.Tensor, "pts ... 2"]:
        pts = self.discretize_pts(pts)
        l = pts.floor()
        wb = pts - l
        wa = 1.0 - wb
        return torch.stack((wa, wb), dim=-1)

    def discretize_pts(
        self, pts: Float[torch.Tensor, "pts dim"]
    ) -> Float[torch.Tensor, "pts dim"]:
        pts = pts * (self.grid_resolution - 2)
        return pts

    def get_data_for_point(
        self,
        point: Float[torch.Tensor, "pts dim"],
        grid: Optional[Float[torch.Tensor, "num_latents dim"]] = None,
    ) -> Float[torch.Tensor, "pts feat_dim"]:
        if grid is None:
            grid = self.latents.weight

        clamp_x = lambda x: x.clamp(0, self.grid_resolution - 1)
        clamp_y = lambda x: x.clamp(0, self.grid_resolution - 1)
        clamp_z = lambda x: x.clamp(0, self.grid_resolution - 1)
        if torch.is_tensor(point):
            p_x, p_y, p_z = point.unbind(-1)
        else:
            p_x, p_y, p_z = point
        p_x = p_x.long()
        p_y = p_y.long()
        p_z = p_z.long()
        grid = grid.view(
            [
                self.grid_resolution,
                self.grid_resolution,
                self.grid_resolution,
                -1,
            ]
        )

        data = grid[clamp_x(p_x), clamp_y(p_y), clamp_z(p_z)]

        return data

    def init_coordinate_transform(self, coords: Float[torch.Tensor, "pts 3"]):
        return coords


class CompositeEncoding(nn.Module):
    def __init__(
        self, encoding, include_xyz=False, xyz_scale=1.0, xyz_offset=0.0
    ):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = (
            include_xyz,
            xyz_scale,
            xyz_offset,
        )
        self.n_output_dims = (
            int(self.include_xyz) * self.encoding.n_input_dims
            + self.encoding.n_output_dims
        )

    def forward(self, x, *args):
        return (
            self.encoding(x, *args)
            if not self.include_xyz
            else torch.cat(
                [
                    x * self.xyz_scale + self.xyz_offset,
                    self.encoding(x, *args),
                ],
                dim=-1,
            )
        )

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)


def get_encoding(n_input_dims, config):
    # input suppose to be range [0, 1]
    if config.otype == "VanillaFrequency":
        encoding = VanillaFrequency(n_input_dims, config_to_primitive(config))
    elif config.otype == "ProgressiveBandHashGrid":
        encoding = ProgressiveBandHashGrid(
            n_input_dims, config_to_primitive(config)
        )
    elif config.otype == "CCEncoding":
        encoding = CCEncoding(n_input_dims, config_to_primitive(config))
    elif config.otype == "BCCEncoding":
        encoding = BCCEncoding(n_input_dims, config_to_primitive(config))
    else:
        with torch.cuda.device(get_rank()):
            encoding = tcnn.Encoding(n_input_dims, config_to_primitive(config))
    encoding = CompositeEncoding(
        encoding,
        include_xyz=config.get("include_xyz", False),
        xyz_scale=2.0,
        xyz_offset=-1.0,
    )
    return encoding


class VanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        self.sphere_init, self.weight_norm = config.get(
            "sphere_init", False
        ), config.get("weight_norm", False)

        self.small_init = config.get("small_init", False)
        self.sphere_init_radius = config.get("sphere_init_radius", 0.5)
        self.layers = [
            self.make_linear(
                dim_in, self.n_neurons, is_first=True, is_last=False
            ),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [
                self.make_linear(
                    self.n_neurons,
                    self.n_neurons,
                    is_first=False,
                    is_last=False,
                ),
                self.make_activation(),
            ]
        self.layers += [
            self.make_linear(
                self.n_neurons, dim_out, is_first=False, is_last=True
            )
        ]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config["output_activation"])

    @torch.cuda.amp.autocast(False)
    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(
            dim_in, dim_out, bias=True
        )  # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(
                    layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out)
                )
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(
                    layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out)
                )
        elif self.small_init and is_last:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.uniform_(layer.weight, -1e-4, 1e-4)
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)


def sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network):
    rank_zero_debug("Initialize tcnn MLP to approximately represent a sphere.")
    """
    from https://github.com/NVlabs/tiny-cuda-nn/issues/96
    It's the weight matrices of each layer laid out in row-major order and then concatenated.
    Notably: inputs and output dimensions are padded to multiples of 8 (CutlassMLP) or 16 (FullyFusedMLP).
    The padded input dimensions get a constant value of 1.0,
    whereas the padded output dimensions are simply ignored,
    so the weights pertaining to those can have any value.
    """
    padto = 16 if config.otype == "FullyFusedMLP" else 8
    n_input_dims = n_input_dims + (padto - n_input_dims % padto) % padto
    n_output_dims = n_output_dims + (padto - n_output_dims % padto) % padto
    data = list(network.parameters())[0].data
    assert (
        data.shape[0]
        == (n_input_dims + n_output_dims) * config.n_neurons
        + (config.n_hidden_layers - 1) * config.n_neurons**2
    )
    new_data = []
    # first layer
    weight = torch.zeros((config.n_neurons, n_input_dims)).to(data)
    torch.nn.init.constant_(weight[:, 3:], 0.0)
    torch.nn.init.normal_(
        weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(config.n_neurons)
    )
    new_data.append(weight.flatten())
    # hidden layers
    for i in range(config.n_hidden_layers - 1):
        weight = torch.zeros((config.n_neurons, config.n_neurons)).to(data)
        torch.nn.init.normal_(
            weight, 0.0, math.sqrt(2) / math.sqrt(config.n_neurons)
        )
        new_data.append(weight.flatten())
    # last layer
    weight = torch.zeros((n_output_dims, config.n_neurons)).to(data)
    torch.nn.init.normal_(
        weight,
        mean=math.sqrt(math.pi) / math.sqrt(config.n_neurons),
        std=0.0001,
    )
    new_data.append(weight.flatten())
    new_data = torch.cat(new_data)
    data.copy_(new_data)


def get_mlp(n_input_dims, n_output_dims, config):
    if config.otype == "VanillaMLP":
        network = VanillaMLP(
            n_input_dims, n_output_dims, config_to_primitive(config)
        )
    elif config.otype == "Identity":
        network = nn.Identity()
    else:
        with torch.cuda.device(get_rank()):
            network = tcnn.Network(
                n_input_dims, n_output_dims, config_to_primitive(config)
            )
            if config.get("sphere_init", False):
                sphere_init_tcnn_network(
                    n_input_dims, n_output_dims, config, network
                )
    return network


class EncodingWithNetwork(nn.Module):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, x):
        return self.network(self.encoding(x))

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)


def get_encoding_with_network(
    n_input_dims, n_output_dims, encoding_config, network_config
):
    # input suppose to be range [0, 1]
    if encoding_config.otype in [
        "VanillaFrequency",
        "ProgressiveBandHashGrid",
    ] or network_config.otype in ["VanillaMLP"]:
        encoding = get_encoding(n_input_dims, encoding_config)
        network = get_mlp(
            encoding.n_output_dims, n_output_dims, network_config
        )
        encoding_with_network = EncodingWithNetwork(encoding, network)
    else:
        with torch.cuda.device(get_rank()):
            encoding_with_network = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=config_to_primitive(encoding_config),
                network_config=config_to_primitive(network_config),
            )
    return encoding_with_network


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter(
            "variance", nn.Parameter(torch.tensor(self.config.init_val))
        )
        self.modulate = self.config.get("modulate", False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s

    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return (
            torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
        )

    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min(
                    (global_step / self.reach_max_steps)
                    * (self.max_inv_s - self.prev_inv_s)
                    + self.prev_inv_s,
                    self.max_inv_s,
                )
