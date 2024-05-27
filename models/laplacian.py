from pathlib import Path
from typing import Callable, Union, cast

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import trimesh
from jaxtyping import Float, Int32, Int64


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


class Laplacian(nn.Module):
    def __init__(
        self, resolution: int, sdf_fun: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.sdf_fun = sdf_fun

        self._create_mesh_structures()

    def _create_mesh_structures(self):
        verts, faces = yet_another_meshing(self.resolution)
        mesh = trimesh.Trimesh(
            vertices=verts.cpu().numpy(),
            faces=faces.cpu().numpy(),
            process=False,
        )

        graph = nx.from_edgelist(mesh.edges_unique)
        two_ring = []
        one_ring = []
        max_len_two_ring = -1
        max_len_one_ring = -1
        for i in range(len(mesh.vertices)):
            ring = list(graph[i].keys())

            single_two_ring = [
                subelem for elem in ring for subelem in graph[elem].keys()
            ]
            two_ring.append(list(sorted(list(set(single_two_ring)))))
            one_ring.append(ring)
            max_len_one_ring = max(max_len_one_ring, len(ring))
            max_len_two_ring = max(max_len_two_ring, len(two_ring[-1]))

        padded_one_ring = []
        padded_two_ring = []
        for o_ring, t_ring in zip(one_ring, two_ring):
            padded_one_ring.append(
                o_ring + [-1] * (max_len_one_ring - len(o_ring))
            )
            padded_two_ring.append(
                t_ring + [-1] * (max_len_two_ring - len(t_ring))
            )
        central_nodes = torch.tensor(list(graph.nodes), dtype=torch.long)
        one_ring_array_indices = torch.tensor(
            padded_one_ring, dtype=torch.long
        )
        two_ring_array_indices = torch.tensor(
            padded_two_ring, dtype=torch.long
        )
        verts_one_ring = verts[one_ring_array_indices]
        verts_two_ring = verts[two_ring_array_indices]
        verts_central_nodes = verts[central_nodes]

        self.register_buffer("verts", verts)
        self.register_buffer("one_ring_array_indices", one_ring_array_indices)
        self.register_buffer("two_ring_array_indices", two_ring_array_indices)

        self.register_buffer("central_nodes", central_nodes)
        self.register_buffer(
            "verts_one_ring", verts_one_ring / (self.resolution - 1)
        )
        self.register_buffer(
            "verts_two_ring", verts_two_ring / (self.resolution - 1)
        )
        self.register_buffer(
            "verts_central_nodes", verts_central_nodes / (self.resolution - 1)
        )
        self.register_buffer(
            "one_ring_mask",
            ((one_ring_array_indices != -1).float().unsqueeze(dim=-1)),
        )

    def forward(self):
        sdf, lap = self.get_laplacian_of_sdfs_and_sdfs()
        return sdf, lap

    def get_laplacian_of_sdfs_and_sdfs(self):
        if torch.is_tensor(step):
            if step.ndim == 0:
                step = step.item()
            else:
                step = step[0].item()

        with torch.no_grad():
            if (
                self.verts is None
                or self.one_ring_array_indices is None
                or self.two_ring_array_indices is None
            ):
                self._create_mesh_structures()
            verts_one_ring = self.verts_one_ring
            verts_central_nodes = self.verts_central_nodes
            one_ring_mask = self.one_ring_mask

        def extract_from_coords(
            coords: Float[torch.Tensor, "... 3"],
        ) -> Float[torch.Tensor, "... dim"]:
            coords_ = coords.view((-1, 3))
            sdfs = self.sdf_fun(coords_)
            sdfs = sdfs.view((*coords.shape[:-1],))
            return sdfs

        one_ring_sdfs = extract_from_coords(verts_one_ring)
        center_sdfs = extract_from_coords(verts_central_nodes)

        lap = (one_ring_sdfs * one_ring_mask).sum(dim=-2) - one_ring_mask.sum(
            dim=-2
        ) * center_sdfs
        return center_sdfs, lap
