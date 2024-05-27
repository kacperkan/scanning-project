import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import trimesh.repair
from jaxtyping import Int64
from pytorch_lightning.utilities.rank_zero import rank_zero_info

import models
from models.base import BaseModel
from models.network_utils import (
    get_encoding,
    get_encoding_with_network,
    get_mlp,
)
from models.utils import (
    ContractionType,
    Features,
    bcc_mesh,
    chunk_batch,
    cleanup,
    extract_voxel_params,
    get_activation,
    scale_anything,
)
from systems.utils import update_module_step
from utils.misc import get_rank


def sort_edges(
    edges_ex2: Int64[torch.Tensor, "num_edges 2"],
) -> Int64[torch.Tensor, "num_edges 2"]:
    with torch.no_grad():
        order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
        order = order.unsqueeze(dim=1)

        a = torch.gather(input=edges_ex2, index=order, dim=1)
        b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

    return torch.stack([a, b], -1)


def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x


class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes

            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes

            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0.0):
        level = level.float().view(
            self.resolution, self.resolution, self.resolution
        )
        if self.use_torch:
            verts, faces = self.mc_func(level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(
                -level.numpy(), threshold
            )  # transform to numpy
            verts, faces = torch.from_numpy(
                verts.astype(np.float32)
            ), torch.from_numpy(
                faces.astype(np.int64)
            )  # transform back to pytorch
        verts = verts / (self.resolution - 1.0)
        return {"v_pos": verts, "t_pos_idx": faces}


class BCCMarchingTetrahedraHelper(nn.Module):
    def __init__(self, resolution, **kwargs) -> None:
        super().__init__()
        self.resolution = resolution
        self.points_range = (0, 1)
        self.verts = None
        self.edges = None
        self.chunk_size = 16384
        self.base_tet_edges = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long
        )
        self.triangle_table = torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1],
                [1, 0, 2, -1, -1, -1],
                [4, 0, 3, -1, -1, -1],
                [1, 4, 2, 1, 3, 4],
                [3, 1, 5, -1, -1, -1],
                [2, 3, 0, 2, 5, 3],
                [1, 4, 0, 1, 5, 4],
                [4, 2, 5, -1, -1, -1],
                [4, 5, 2, -1, -1, -1],
                [4, 1, 0, 4, 5, 1],
                [3, 2, 0, 3, 5, 2],
                [1, 3, 5, -1, -1, -1],
                [4, 1, 2, 4, 3, 1],
                [3, 0, 4, -1, -1, -1],
                [2, 0, 1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.long,
        )
        self.num_triangles_table = torch.tensor(
            [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0],
            dtype=torch.long,
        )

    def grid_vertices(self):
        if self.verts is None:
            verts, edges = [], []
            for v, e in bcc_mesh(self.resolution, self.chunk_size):
                verts.append(v)
                edges.append(e)

            verts = torch.cat(verts, dim=0)
            edges = torch.cat(edges, dim=0)

            self.edges = edges.view((-1, 4, 3))

            verts = verts.view((-1, verts.shape[-1]))
            verts = verts * torch.tensor((0.5, 0.5, 1.0)).to(verts)
            verts = verts.floor().clamp(0, self.resolution)
            verts = verts / verts.max()

            self.verts = verts

        return self.verts

    def forward(self, sdf_n, threshold=0.0):
        edges = self.edges.to(sdf_n.device)
        pos_nx3 = self.verts.to(sdf_n.device)
        self.base_tet_edges.to(sdf_n.device)
        self.num_triangles_table.to(sdf_n.device)
        self.triangle_table.to(sdf_n.device)
        with torch.no_grad():
            occ_n = sdf_n > threshold
            occ_fx4 = occ_n.view((-1, 4))
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = edges[valid_tets]
            all_edges = all_edges[:, 0, 0]  # get coordinate of the tet
            all_edges = torch.stack(
                [all_edges, all_edges + 1, all_edges + 2, all_edges + 3], dim=1
            )
            all_edges = all_edges[:, self.base_tet_edges].reshape((-1, 2))

            all_edges = sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(
                all_edges, dim=0, return_inverse=True
            )

            unique_edges = unique_edges.long()
            mask_edges = (
                occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            )
            mapping = (
                torch.ones(
                    (unique_edges.shape[0]),
                    dtype=torch.long,
                    device=sdf_n.device,
                )
                * -1
            )
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device=sdf_n.device
            )
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = (
            torch.flip(edges_to_interp_sdf, [1]) / denominator
        )
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(
            2, torch.arange(4, dtype=torch.long, device=sdf_n.device)
        )
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][
                        :, :3
                    ],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][
                        :, :6
                    ],
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        mesh = trimesh.Trimesh(vertices=verts.cpu(), faces=faces.cpu())
        trimesh.repair.fix_winding(mesh)
        mesh.invert()
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        return {"v_pos": verts, "t_pos_idx": faces}


class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in [
                "mc",
                "mc-torch",
                "bcc-mt",
            ]
            if self.config.isosurface.method == "mc-torch":
                raise NotImplementedError(
                    "Please do not use mc-torch. It currently has some scaling"
                    " issues I haven't fixed yet."
                )

            if self.config.isosurface.method == "bcc-mt":
                self.helper = BCCMarchingTetrahedraHelper(
                    self.config.isosurface.resolution
                )
            else:
                self.helper = MarchingCubeHelper(
                    self.config.isosurface.resolution,
                    use_torch=self.config.isosurface.method == "mc-torch",
                )
        self.radius = self.config.radius
        self.contraction_type = None  # assigned in system

    def forward_level(self, points):
        raise NotImplementedError

    def isosurface_(self, vmin, vmax):
        verts = self.helper.grid_vertices()

        def batch_func(x):
            x = torch.stack(
                [
                    scale_anything(x[..., 0], (0, 1), (vmin[0], vmax[0])),
                    scale_anything(x[..., 1], (0, 1), (vmin[1], vmax[1])),
                    scale_anything(x[..., 2], (0, 1), (vmin[2], vmax[2])),
                ],
                dim=-1,
            ).to(self.rank)
            rv = self.forward_level(x).cpu()
            cleanup()
            return rv

        level = chunk_batch(
            batch_func, self.config.isosurface.chunk, True, verts
        )
        mesh = self.helper(level, threshold=self.config.isosurface.threshold)
        mesh["v_pos"] = torch.stack(
            [
                scale_anything(
                    mesh["v_pos"][..., 0], (0, 1), (vmin[0], vmax[0])
                ),
                scale_anything(
                    mesh["v_pos"][..., 1], (0, 1), (vmin[1], vmax[1])
                ),
                scale_anything(
                    mesh["v_pos"][..., 2], (0, 1), (vmin[2], vmax[2])
                ),
            ],
            dim=-1,
        )
        return mesh

    @torch.no_grad()
    def isosurface(self):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse = self.isosurface_(
            (-self.radius, -self.radius, -self.radius),
            (self.radius, self.radius, self.radius),
        )
        vmin, vmax = mesh_coarse["v_pos"].amin(dim=0), mesh_coarse[
            "v_pos"
        ].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        mesh_fine = self.isosurface_(vmin_, vmax_)
        return mesh_fine


@models.register("volume-density")
class VolumeDensity(BaseImplicitGeometry):
    def setup(self):
        self.n_input_dims = self.config.get("n_input_dims", 3)
        self.n_output_dims = self.config.feature_dim
        self.encoding_with_network = get_encoding_with_network(
            self.n_input_dims,
            self.n_output_dims,
            self.config.xyz_encoding_config,
            self.config.mlp_network_config,
        )

    def forward(self, points):
        points = contract_to_unisphere(
            points, self.radius, self.contraction_type
        )
        out = (
            self.encoding_with_network(points.view(-1, self.n_input_dims))
            .view(*points.shape[:-1], self.n_output_dims)
            .float()
        )
        density, feature = out[..., 0], out
        if "density_activation" in self.config:
            density = get_activation(self.config.density_activation)(
                density + float(self.config.density_bias)
            )
        if "feature_activation" in self.config:
            feature = get_activation(self.config.feature_activation)(feature)
        return density, feature

    def forward_level(self, points):
        points = contract_to_unisphere(
            points, self.radius, self.contraction_type
        )
        density = self.encoding_with_network(
            points.reshape(-1, self.n_input_dims)
        ).reshape(*points.shape[:-1], self.n_output_dims)[..., 0]
        if "density_activation" in self.config:
            density = get_activation(self.config.density_activation)(
                density + float(self.config.density_bias)
            )
        return -density

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding_with_network, epoch, global_step)


@models.register("volume-sdf")
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(
            encoding.n_output_dims,
            self.n_output_dims,
            self.config.mlp_network_config,
        )
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get(
            "finite_difference_eps", 1e-3
        )
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        if self.grad_type == "finite_difference":
            rank_zero_info(
                "Using finite difference to compute gradients with"
                f" eps={self.finite_difference_eps}"
            )

    def forward(
        self, points, with_grad=True, with_feature=True, with_laplace=False
    ):
        with torch.inference_mode(
            torch.is_inference_mode_enabled()
            and not (with_grad and self.grad_type == "analytic")
        ):
            with torch.set_grad_enabled(
                self.training or (with_grad and self.grad_type == "analytic")
            ):
                if with_grad and self.grad_type == "analytic":
                    if not self.training:
                        points = (
                            points.clone()
                        )  # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points  # points in the original scale
                points = contract_to_unisphere(
                    points, self.radius, self.contraction_type
                )  # points normalized to (0, 1)

                out = (
                    self.network(self.encoding(points.view(-1, 3)))
                    .view(*points.shape[:-1], self.n_output_dims)
                    .float()
                )
                sdf, feature = out[..., 0], out
                if "sdf_activation" in self.config:
                    sdf = get_activation(self.config.sdf_activation)(
                        sdf + float(self.config.sdf_bias)
                    )
                if "feature_activation" in self.config:
                    feature = get_activation(self.config.feature_activation)(
                        feature
                    )
                if with_grad:
                    if self.grad_type == "analytic":
                        grad = torch.autograd.grad(
                            sdf,
                            points_,
                            grad_outputs=torch.ones_like(sdf),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                    elif self.grad_type == "finite_difference":
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[..., None, :] + offsets).clamp(
                            -self.radius, self.radius
                        )
                        points_d = scale_anything(
                            points_d_, (-self.radius, self.radius), (0, 1)
                        )
                        points_d_sdf = (
                            self.network(self.encoding(points_d.view(-1, 3)))[
                                ..., 0
                            ]
                            .view(*points.shape[:-1], 6)
                            .float()
                        )
                        grad = (
                            0.5
                            * (
                                points_d_sdf[..., 0::2]
                                - points_d_sdf[..., 1::2]
                            )
                            / eps
                        )

                        if with_laplace:
                            laplace = (
                                points_d_sdf[..., 0::2]
                                + points_d_sdf[..., 1::2]
                                - 2 * sdf[..., None]
                            ).sum(-1) / (eps**2)

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            assert self.config.grad_type == "finite_difference", (
                "Laplace computation is only supported with"
                " grad_type='finite_difference'"
            )
            rv.append(laplace)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = contract_to_unisphere(
            points, self.radius, self.contraction_type
        )  # points normalized to (0, 1)
        sdf = self.network(self.encoding(points.view(-1, 3))).view(
            *points.shape[:-1], self.n_output_dims
        )[..., 0]
        if "sdf_activation" in self.config:
            sdf = get_activation(self.config.sdf_activation)(
                sdf + float(self.config.sdf_bias)
            )
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)
        if self.grad_type == "finite_difference":
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == "progressive":
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", (
                    "finite_difference_eps='progressive' only works with"
                    " ProgressiveBandHashGrid"
                )
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0)
                    // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = (
                    hg_conf.base_resolution
                    * hg_conf.per_level_scale ** (current_level - 1)
                )
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(
                        f"Update finite_difference_eps to {grid_size}"
                    )
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(
                    "Unknown"
                    f" finite_difference_eps={self.finite_difference_eps}"
                )


@models.register("bcc-sdf")
class BCCSDF(BaseImplicitGeometry):
    def setup(self):
        encoding = get_encoding(3, self.config.xyz_encoding_config)

        self.feats_to_encode = [
            Features.from_str(feat)
            for feat in self.config.xyz_encoding_config.feats_to_encode
        ]
        self.is_direct_encoding = (
            self.config.xyz_encoding_config.stored_data == "direct"
        )

        if self.is_direct_encoding:
            if Features.LATENT_FEATS in self.feats_to_encode:
                self.n_output_dims = Features.LATENT_FEATS.dim
            else:
                self.n_output_dims = encoding.n_output_dims
            self.mlp_output_dims = encoding.n_output_dims
        else:
            self.mlp_output_dims = self.n_output_dims = self.config.feature_dim

        network = get_mlp(
            encoding.n_output_dims,
            self.mlp_output_dims,
            self.config.mlp_network_config,
        )
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get(
            "finite_difference_eps", 1e-3
        )
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        if self.grad_type == "finite_difference":
            rank_zero_info(
                "Using finite difference to compute gradients with"
                f" eps={self.finite_difference_eps}"
            )

    def forward(
        self, points, with_grad=True, with_feature=True, with_laplace=False
    ):
        with torch.inference_mode(
            torch.is_inference_mode_enabled()
            and not (with_grad and self.grad_type == "analytic")
        ):
            with torch.set_grad_enabled(
                self.training or (with_grad and self.grad_type == "analytic")
            ):
                if with_grad and self.grad_type == "analytic":
                    if not self.training:
                        points = (
                            points.clone()
                        )  # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points  # points in the original scale
                points = contract_to_unisphere(
                    points, self.radius, self.contraction_type
                )  # points normalized to (0, 1)

                out = (
                    self.network(self.encoding(points.view(-1, 3)))
                    .view(*points.shape[:-1], self.mlp_output_dims)
                    .float()
                )
                params_out = extract_voxel_params(
                    out,
                    self.feats_to_encode,
                    self.config.xyz_encoding_config.rgb_basis_dim,
                )
                sdf = params_out[Features.DENSITY][..., 0]
                feature = params_out[Features.LATENT_FEATS]
                if "sdf_activation" in self.config:
                    sdf = get_activation(self.config.sdf_activation)(
                        sdf + float(self.config.sdf_bias)
                    )
                if "feature_activation" in self.config:
                    feature = get_activation(self.config.feature_activation)(
                        feature
                    )
                if with_grad:
                    if self.grad_type == "analytic":
                        grad = torch.autograd.grad(
                            sdf,
                            points_,
                            grad_outputs=torch.ones_like(sdf),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                    elif self.grad_type == "finite_difference":
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[..., None, :] + offsets).clamp(
                            -self.radius, self.radius
                        )
                        points_d = scale_anything(
                            points_d_, (-self.radius, self.radius), (0, 1)
                        )
                        out_d = self.network(
                            self.encoding(points_d.view(-1, 3))
                        )
                        params_d_out = extract_voxel_params(
                            out_d,
                            self.feats_to_encode,
                            self.config.xyz_encoding_config.rgb_basis_dim,
                        )
                        points_d_sdf = (
                            params_d_out[Features.DENSITY][..., 0]
                            .view(*points.shape[:-1], 6)
                            .float()
                        )
                        grad = (
                            0.5
                            * (
                                points_d_sdf[..., 0::2]
                                - points_d_sdf[..., 1::2]
                            )
                            / eps
                        )

                        if with_laplace:
                            laplace = (
                                points_d_sdf[..., 0::2]
                                + points_d_sdf[..., 1::2]
                                - 2 * sdf[..., None]
                            ).sum(-1) / (eps**2)

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            assert self.config.grad_type == "finite_difference", (
                "Laplace computation is only supported with"
                " grad_type='finite_difference'"
            )
            rv.append(laplace)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = contract_to_unisphere(
            points, self.radius, self.contraction_type
        )  # points normalized to (0, 1)
        out = self.network(self.encoding(points.view(-1, 3))).view(
            *points.shape[:-1], self.mlp_output_dims
        )
        params_out = extract_voxel_params(
            out,
            self.feats_to_encode,
            self.config.xyz_encoding_config.rgb_basis_dim,
        )
        sdf = params_out[Features.DENSITY][..., 0]
        if "sdf_activation" in self.config:
            sdf = get_activation(self.config.sdf_activation)(
                sdf + float(self.config.sdf_bias)
            )
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)
        if self.grad_type == "finite_difference":
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == "progressive":
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", (
                    "finite_difference_eps='progressive' only works with"
                    " ProgressiveBandHashGrid"
                )
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0)
                    // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = (
                    hg_conf.base_resolution
                    * hg_conf.per_level_scale ** (current_level - 1)
                )
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(
                        f"Update finite_difference_eps to {grid_size}"
                    )
                self._finite_difference_eps = grid_size
            elif self.finite_difference_eps == "from_res":
                hg_conf = self.config.xyz_encoding_config
                grid_res = hg_conf.grid_resolution
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(
                        f"Update finite_difference_eps to {grid_size}"
                    )
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(
                    "Unknown"
                    f" finite_difference_eps={self.finite_difference_eps}"
                )


@models.register("bcc-sdf-with-aux")
class BCCSDFWithAux(BaseImplicitGeometry):
    def setup(self):
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        aux_encoding = get_encoding(3, self.config.aux_xyz_encoding_config)

        self.feats_to_encode = [
            Features.from_str(feat)
            for feat in self.config.xyz_encoding_config.feats_to_encode
        ]
        self.is_direct_encoding = (
            self.config.xyz_encoding_config.stored_data == "direct"
        )

        if self.is_direct_encoding:
            if Features.LATENT_FEATS in self.feats_to_encode:
                self.n_output_dims = Features.LATENT_FEATS.dim
            else:
                self.n_output_dims = encoding.n_output_dims
            self.mlp_output_dims = encoding.n_output_dims
        else:
            self.mlp_output_dims = self.n_output_dims = self.config.feature_dim

        network = get_mlp(
            encoding.n_output_dims,
            self.mlp_output_dims,
            self.config.mlp_network_config,
        )
        aux_network = get_mlp(
            aux_encoding.n_output_dims,
            self.mlp_output_dims,
            self.config.aux_mlp_network_config,
        )

        self.encoding, self.network = encoding, network
        self.aux_encoding, self.aux_network = aux_encoding, aux_network

        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get(
            "finite_difference_eps", 1e-3
        )
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        if self.grad_type == "finite_difference":
            rank_zero_info(
                "Using finite difference to compute gradients with"
                f" eps={self.finite_difference_eps}"
            )

        # stored for the regularizations
        self.__last_aux_sdf = None
        self.__last_aux_features = None

    def forward(
        self, points, with_grad=True, with_feature=True, with_laplace=False
    ):
        with torch.inference_mode(
            torch.is_inference_mode_enabled()
            and not (with_grad and self.grad_type == "analytic")
        ):
            with torch.set_grad_enabled(
                self.training or (with_grad and self.grad_type == "analytic")
            ):
                if with_grad and self.grad_type == "analytic":
                    if not self.training:
                        points = (
                            points.clone()
                        )  # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points  # points in the original scale
                points = contract_to_unisphere(
                    points, self.radius, self.contraction_type
                )  # points normalized to (0, 1)

                sdf, feature = self._get_outputs(
                    points, self.network, self.encoding
                )
                aux_sdf, aux_feature = self._get_outputs(
                    points, self.aux_network, self.aux_encoding
                )

                feature = feature * (aux_feature + 1)

                if "sdf_activation" in self.config:
                    sdf = get_activation(self.config.sdf_activation)(
                        sdf + float(self.config.sdf_bias)
                    )
                if "feature_activation" in self.config:
                    feature = get_activation(self.config.feature_activation)(
                        feature
                    )
                if with_grad:
                    if self.grad_type == "analytic":
                        raise ValueError("Analytic gradient not supported")
                    elif self.grad_type == "finite_difference":
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[..., None, :] + offsets).clamp(
                            -self.radius, self.radius
                        )
                        points_d = scale_anything(
                            points_d_, (-self.radius, self.radius), (0, 1)
                        )
                        d_sdf, _ = self._get_outputs(
                            points_d, self.network, self.encoding
                        )
                        d_aux_sdf, _ = self._get_outputs(
                            points_d, self.aux_network, self.aux_encoding
                        )
                        d_sdf = d_sdf + d_aux_sdf
                        points_d_sdf = d_sdf.view(
                            *points.shape[:-1], 6
                        ).float()
                        grad = (
                            0.5
                            * (
                                points_d_sdf[..., 0::2]
                                - points_d_sdf[..., 1::2]
                            )
                            / eps
                        )

                        if with_laplace:
                            laplace = (
                                points_d_sdf[..., 0::2]
                                + points_d_sdf[..., 1::2]
                                - 2 * sdf[..., None]
                            ).sum(-1) / (eps**2)

        if self.training:
            self.__last_aux_sdf = aux_sdf
            self.__last_aux_features = aux_feature
        else:
            self.__last_aux_sdf = None
            self.__last_aux_features = None

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            assert self.config.grad_type == "finite_difference", (
                "Laplace computation is only supported with"
                " grad_type='finite_difference'"
            )
            rv.append(laplace)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = contract_to_unisphere(
            points, self.radius, self.contraction_type
        )  # points normalized to (0, 1)
        sdf = self._get_outputs(points, self.network, self.encoding)[0]
        aux_sdf = self._get_outputs(
            points, self.aux_network, self.aux_encoding
        )[0]
        sdf = sdf + aux_sdf

        if "sdf_activation" in self.config:
            sdf = get_activation(self.config.sdf_activation)(
                sdf + float(self.config.sdf_bias)
            )
        return sdf

    def _get_outputs(self, points, network, encoding):
        out = (
            network(encoding(points.view(-1, 3)))
            .view(*points.shape[:-1], self.mlp_output_dims)
            .float()
        )
        params_out = extract_voxel_params(
            out,
            self.feats_to_encode,
            self.config.xyz_encoding_config.rgb_basis_dim,
        )
        sdf = params_out[Features.DENSITY][..., 0]
        feature = params_out[Features.LATENT_FEATS]
        return sdf, feature

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)
        if self.grad_type == "finite_difference":
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == "progressive":
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", (
                    "finite_difference_eps='progressive' only works with"
                    " ProgressiveBandHashGrid"
                )
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0)
                    // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = (
                    hg_conf.base_resolution
                    * hg_conf.per_level_scale ** (current_level - 1)
                )
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(
                        f"Update finite_difference_eps to {grid_size}"
                    )
                self._finite_difference_eps = grid_size
            elif self.finite_difference_eps == "from_res":
                hg_conf = self.config.xyz_encoding_config
                grid_res = hg_conf.grid_resolution
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(
                        f"Update finite_difference_eps to {grid_size}"
                    )
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(
                    "Unknown"
                    f" finite_difference_eps={self.finite_difference_eps}"
                )

    def regularizations(self, out):
        aux_sdf = self.__last_aux_sdf
        self.__last_aux_sdf = None

        aux_feature = self.__last_aux_features
        self.__last_aux_features = None

        aux_sdf_reg = (
            aux_sdf.view(-1).abs().mean()
            + aux_feature.abs().sum(dim=-1).mean()
        )

        return {"aux_sdf_reg": aux_sdf_reg}
