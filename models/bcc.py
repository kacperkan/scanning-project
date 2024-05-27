import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfacc import (
    OccGridEstimator,
    accumulate_along_rays,
    render_weight_from_alpha,
    render_weight_from_density,
)
from nerfacc.grid import ray_aabb_intersect

import models
from models.base import BaseModel
from models.network_utils import VarianceNetwork
from models.utils import ContractionType, chunk_batch
from systems.utils import update_module_step


@models.register("bcc")
class BCCModel(BaseModel):
    def setup(self):
        self.geometry = models.make(
            self.config.geometry.name, self.config.geometry
        )
        self.texture = models.make(
            self.config.texture.name, self.config.texture
        )
        self.geometry.contraction_type = ContractionType.AABB

        if self.config.learned_background:
            self.geometry_bg = models.make(
                self.config.geometry_bg.name, self.config.geometry_bg
            )
            self.texture_bg = models.make(
                self.config.texture_bg.name, self.config.texture_bg
            )
            self.geometry_bg.contraction_type = (
                ContractionType.UN_BOUNDED_SPHERE
            )
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = (
                10
                ** (
                    math.log10(self.far_plane_bg)
                    / self.config.num_samples_per_ray_bg
                )
                - 1.0
            )
            self.render_step_size_bg = 0.01

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer(
            "scene_aabb",
            torch.as_tensor(
                [
                    -self.config.radius,
                    -self.config.radius,
                    -self.config.radius,
                    self.config.radius,
                    self.config.radius,
                    self.config.radius,
                ],
                dtype=torch.float32,
            ),
        )
        if self.config.grid_prune:
            self.occupancy_grid = OccGridEstimator(
                roi_aabb=self.scene_aabb,
                resolution=128,
            )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccGridEstimator(
                    roi_aabb=self.scene_aabb, resolution=256
                )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = (
            1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
        )

    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get("cos_anneal_end", 0)
        self.cos_anneal_ratio = (
            1.0
            if cos_anneal_end == 0
            else min(1.0, global_step / cos_anneal_end)
        )

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[..., None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[..., None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha

        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[..., None] * self.render_step_size_bg

        if self.training and self.config.grid_prune:
            self.occupancy_grid.update_every_n_steps(
                step=global_step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=self.config.get("grid_prune_occ_thre", 0.01),
            )
            if self.config.learned_background:
                self.occupancy_grid_bg.update_every_n_steps(
                    step=global_step,
                    occ_eval_fn=occ_eval_fn_bg,
                    occ_thre=self.config.get("grid_prune_occ_thre_bg", 0.01),
                )

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(
            1e-6, 1e6
        )  # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio)
            + F.relu(-true_cos) * self.cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = (
            sdf[..., None] + iter_cos * dists.reshape(-1, 1) * 0.5
        )
        estimated_prev_sdf = (
            sdf[..., None] - iter_cos * dists.reshape(-1, 1) * 0.5
        )

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_bg_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = (
                t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            )
            density, _ = self.geometry_bg(positions)
            return density

        _, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        # if the ray intersects with the bounding box, start from the farther intersection point
        # otherwise start from self.far_plane_bg
        # note that in nerfacc t_max is set to 1e10 if there is no intersection
        near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)
        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.occupancy_grid_bg.sampling(
                rays_o,
                rays_d,
                sigma_fn=sigma_fn,
                near_plane=near_plane if near_plane else 0.0,
                far_plane=self.far_plane_bg if self.far_plane_bg else 1e8,
                render_step_size=self.render_step_size_bg,
                stratified=self.randomized,
                cone_angle=self.cone_angle_bg,
                alpha_thre=0.0,
            )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends)[..., None] / 2.0
        positions = t_origins + t_dirs * midpoints
        intervals = (t_ends - t_starts)[..., None]

        density, feature = self.geometry_bg(positions)
        rgb = self.texture_bg(feature, t_dirs)

        weights, _, _ = render_weight_from_density(
            t_starts, t_ends, density, ray_indices=ray_indices, n_rays=n_rays
        )
        opacity = accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=None,
            n_rays=n_rays,
        )
        depth = accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=midpoints,
            n_rays=n_rays,
        )
        comp_rgb = accumulate_along_rays(
            weights=weights, ray_indices=ray_indices, values=rgb, n_rays=n_rays
        )
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb,
            "opacity": opacity,
            "depth": depth,
            "rays_valid": opacity > 0,
            "num_samples": torch.as_tensor(
                [len(t_starts)], dtype=torch.int32, device=rays.device
            ),
        }

        if self.training:
            out.update(
                {
                    "weights": weights.view(-1),
                    "points": midpoints.view(-1),
                    "intervals": intervals.view(-1),
                    "ray_indices": ray_indices.view(-1),
                }
            )

        return out

    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.occupancy_grid.sampling(
                rays_o,
                rays_d,
                alpha_fn=None,
                near_plane=0.0,
                far_plane=1e8,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0,
            )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends)[..., None] / 2.0
        positions = t_origins + t_dirs * midpoints
        dists = (t_ends - t_starts)[..., None]

        if self.config.geometry.grad_type == "finite_difference":
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(
                positions, with_grad=True, with_feature=True, with_laplace=True
            )
        else:
            sdf, sdf_grad, feature = self.geometry(
                positions, with_grad=True, with_feature=True
            )
        eps = torch.finfo(sdf.dtype).eps
        where_all_zeros = torch.where(sdf_grad.abs().sum(-1) < eps)
        sdf_grad[where_all_zeros] = eps
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)
        # rgb = self.texture(positions, feature, t_dirs, normal)
        rgb = self.texture(
            positions,
            feature,
            torch.zeros_like(t_dirs),
            torch.zeros_like(normal),
        )

        weights, _ = render_weight_from_alpha(
            alpha, ray_indices=ray_indices, n_rays=n_rays
        )
        opacity = accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=None,
            n_rays=n_rays,
        )
        depth = accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=midpoints,
            n_rays=n_rays,
        )
        comp_rgb = accumulate_along_rays(
            weights=weights, ray_indices=ray_indices, values=rgb, n_rays=n_rays
        )

        comp_normal = accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=normal,
            n_rays=n_rays,
        )
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        out = {
            "comp_rgb": comp_rgb,
            "comp_normal": comp_normal,
            "opacity": opacity,
            "depth": depth,
            "rays_valid": opacity > 0,
            "num_samples": torch.as_tensor(
                [len(t_starts)], dtype=torch.int32, device=rays.device
            ),
        }

        if self.training:
            out.update(
                {
                    "sdf_samples": sdf,
                    "sdf_grad_samples": sdf_grad,
                    "weights": weights.view(-1),
                    "points": midpoints.view(-1),
                    "intervals": dists.view(-1),
                    "ray_indices": ray_indices.view(-1),
                }
            )
            if self.config.geometry.grad_type == "finite_difference":
                out.update({"sdf_laplace_samples": sdf_laplace})

        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                "comp_rgb": self.background_color[None, :].expand(
                    *comp_rgb.shape
                ),
                "num_samples": torch.zeros_like(out["num_samples"]),
                "rays_valid": torch.zeros_like(out["rays_valid"]),
            }

        out_full = {
            "comp_rgb": out["comp_rgb"]
            + out_bg["comp_rgb"] * (1.0 - out["opacity"]),
            "num_samples": out["num_samples"] + out_bg["num_samples"],
            "rays_valid": out["rays_valid"] | out_bg["rays_valid"],
        }

        return {
            **out,
            **{k + "_bg": v for k, v in out_bg.items()},
            **{k + "_full": v for k, v in out_full.items()},
        }

    def forward(self, rays):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
        return {**out, "inv_s": self.variance.inv_s}

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()

    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            pos = mesh["v_pos"].to(self.rank)
            _, sdf_grad, feature = chunk_batch(
                self.geometry,
                export_config.chunk_size,
                False,
                pos,
                with_grad=True,
                with_feature=True,
            )
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            rgb = self.texture(
                pos, feature, -normal, normal
            )  # set the viewing directions to the normal to get "albedo"
            mesh["v_rgb"] = rgb.cpu()
        return mesh
