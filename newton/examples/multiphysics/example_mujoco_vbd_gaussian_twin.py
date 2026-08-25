# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid-VBD Coupled Solver with a Gaussian Deformable Twin
#
# A plush toy captured as a 3D Gaussian splat asset is simulated as a
# volumetric soft body: a tetrahedral cage is voxelized from the Gaussian
# centers and simulated by VBD, while the Gaussian field itself is
# render-only and skinned to the cage every frame ("deformable twin").
#
# Rigid bodies (a swinging paddle driven by MuJoCo and a falling
# box) knock the toy over.  Contacts are exchanged through
# ``SolverCoupledProxy``, so the rigid solver and VBD see each other.
#
# The Gaussian field carries no mass and no collision geometry; the tet
# cage owns all physical state.  Each Gaussian is bound once to the
# bind-pose tetrahedron that contains it, and its center/orientation are
# evaluated from the live cage every frame.
#
# Pass ``--solver vbd`` to run the same scene with a single VBD solver
# (no coupling) as a reference baseline.
#
# Command: python -m newton.examples mujoco_vbd_gaussian_twin
#          python -m newton.examples mujoco_vbd_gaussian_twin --solver vbd
#
###########################################################################

from __future__ import annotations

import argparse
import os

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverMuJoCo, SolverVBD

# The Gaussian splat asset is not redistributed with Newton. Override the path with
# ``--asset`` or the NEWTON_GAUSSIAN_TWIN_ASSET environment variable.
DEFAULT_ASSET = os.environ.get(
    "NEWTON_GAUSSIAN_TWIN_ASSET", "/mnt/data/isaac_lab_poc/ExportedToys/baked.BluehairRagdoll.usdz"
)

# Alternating 5-tetrahedra decompositions of a cube. Corner ``i`` is the
# lattice offset ``(i & 1, (i >> 1) & 1, (i >> 2) & 1)``. Neighboring cells use
# opposite parities so shared faces are triangulated consistently.
_CUBE_TETS_EVEN = np.array([[0, 1, 2, 4], [1, 3, 2, 7], [1, 4, 5, 7], [2, 6, 4, 7], [1, 2, 4, 7]])
_CUBE_TETS_ODD = np.array([[0, 1, 3, 5], [0, 3, 2, 6], [0, 5, 4, 6], [3, 5, 7, 6], [0, 3, 5, 6]])
_CUBE_CORNERS = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)])


def build_tet_cage(points: np.ndarray, cell_size: float, min_cell_points: int):
    """Voxelize *points* and build a tetrahedral cage from the occupied cells.

    Cells holding fewer than *min_cell_points* points are discarded, which drops
    reconstruction outliers instead of growing the cage around them.

    Returns:
        Tuple of ``(vertices, tets, cell_of_point, keep_mask)`` where ``tets`` is
        a ``(num_tets, 4)`` index array with positive-volume orientation and
        ``cell_of_point`` indexes the kept cells for every kept point.
    """
    lower = points.min(axis=0)
    cells = np.floor((points - lower) / cell_size).astype(np.int64)
    occupied, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)

    cell_keep = counts >= min_cell_points
    keep_mask = cell_keep[inverse]
    # Renumber the surviving cells so cell_of_point stays contiguous.
    cell_remap = np.full(len(occupied), -1, dtype=np.int64)
    cell_remap[cell_keep] = np.arange(int(cell_keep.sum()))
    occupied = occupied[cell_keep]
    cell_of_point = cell_remap[inverse[keep_mask]]

    corner_ijk = (occupied[:, None, :] + _CUBE_CORNERS[None, :, :]).reshape(-1, 3)
    vertex_ijk, corner_index = np.unique(corner_ijk, axis=0, return_inverse=True)
    corner_index = corner_index.reshape(-1, 8)
    vertices = lower + vertex_ijk * cell_size

    parity_even = (occupied.sum(axis=1) % 2) == 0
    cube_tets = np.where(parity_even[:, None, None], _CUBE_TETS_EVEN[None], _CUBE_TETS_ODD[None])
    tets = np.take_along_axis(corner_index[:, None, :], cube_tets, axis=2).reshape(-1, 4)

    # Enforce positive volume so the elastic energy sees no inverted rest elements.
    v = vertices[tets]
    volume = np.einsum("ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v[:, 3] - v[:, 0])
    inverted = volume < 0.0
    tets[inverted] = tets[inverted][:, [0, 2, 1, 3]]

    return vertices, tets, cell_of_point, keep_mask


def embed_points_in_cage(points: np.ndarray, vertices: np.ndarray, tets: np.ndarray, cell_of_point: np.ndarray):
    """Bind every point to the cage tetrahedron that contains it.

    Only the five tetrahedra of the point's own cell are candidates, so the
    search is exact and independent of cage size.

    Returns:
        Tuple of ``(tet_of_point, weights)`` with barycentric ``weights`` of
        shape ``(num_points, 4)``.
    """
    candidates = tets.reshape(-1, 5, 4)[cell_of_point]  # (N, 5, 4)
    corners = vertices[candidates]  # (N, 5, 4, 3)
    basis = np.stack(
        [corners[:, :, 1] - corners[:, :, 0], corners[:, :, 2] - corners[:, :, 0], corners[:, :, 3] - corners[:, :, 0]],
        axis=-1,
    )
    rhs = (points[:, None, :] - corners[:, :, 0])[..., None]
    bary = np.linalg.solve(basis, rhs)[..., 0]
    weights = np.concatenate([1.0 - bary.sum(axis=-1, keepdims=True), bary], axis=-1)

    # The containing tetrahedron is the one whose most-negative weight is largest.
    rows = np.arange(len(points))
    best = weights.min(axis=-1).argmax(axis=1)
    return cell_of_point * 5 + best, weights[rows, best]


@wp.kernel
def skin_gaussians_to_tets(
    particle_q: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    gaussian_tet_id: wp.array[wp.int32],
    gaussian_weights: wp.array[wp.vec4],
    rest_rotation: wp.array[wp.quat],
    rest_inverse: wp.array[wp.mat33],
    transforms_out: wp.array[wp.transform],
):
    """Update Gaussian splat centers/rotations from a live VBD tet mesh.

    Each Gaussian carries a static binding to one tetrahedron.  The live
    tet corners are combined with the cached barycentric weights to place
    the Gaussian center.  Orientation follows the deformation-gradient
    rotation of that tet relative to its bind pose.
    """
    i = wp.tid()
    tet = gaussian_tet_id[i]
    if tet < 0:
        return

    w = gaussian_weights[i]
    v0 = particle_q[tet_indices[tet, 0]]
    v1 = particle_q[tet_indices[tet, 1]]
    v2 = particle_q[tet_indices[tet, 2]]
    v3 = particle_q[tet_indices[tet, 3]]

    center = v0 * w[0] + v1 * w[1] + v2 * w[2] + v3 * w[3]

    # Deformation gradient F = live_edges * rest_edges^-1.
    F = wp.matrix_from_cols(v1 - v0, v2 - v0, v3 - v0) * rest_inverse[tet]

    # Rotation part of F, extracted by Gram-Schmidt orthonormalization of its
    # columns. Stretch is intentionally dropped: only Gaussian centers and
    # orientations are skinned, covariance stays at its bind-pose value.
    c0 = wp.normalize(wp.vec3(F[0, 0], F[1, 0], F[2, 0]))
    c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    c1 = wp.normalize(c1 - c0 * wp.dot(c0, c1))
    c2 = wp.cross(c0, c1)

    q_rot = wp.quat_from_matrix(wp.matrix_from_cols(c0, c1, c2))
    transforms_out[i] = wp.transform(center, q_rot * rest_rotation[i])


class GaussianTwin:
    """Static binding between a Gaussian splat asset and a VBD tetrahedral cage."""

    def __init__(self, gaussian: newton.Gaussian, tet_of_gaussian, weights, rest_inverse):
        self.gaussian = gaussian
        self.tet_of_gaussian = wp.array(tet_of_gaussian, dtype=wp.int32)
        self.weights = wp.array(weights, dtype=wp.vec4)
        self.rest_rotation = wp.array(gaussian.rotations, dtype=wp.quat)
        self.rest_inverse = wp.array(rest_inverse, dtype=wp.mat33)

    def update(self, model: newton.Model, state: newton.State):
        """Skin the Gaussian field to the live tet mesh (render-only, no feedback)."""
        wp.launch(
            kernel=skin_gaussians_to_tets,
            dim=self.gaussian.count,
            inputs=[
                state.particle_q,
                model.tet_indices,
                self.tet_of_gaussian,
                self.weights,
                self.rest_rotation,
                self.rest_inverse,
            ],
            outputs=[self.gaussian.warp_data.transforms],
        )


def load_gaussian_asset(path: str) -> newton.Gaussian:
    """Load the first Gaussian splat field found in a USD/USDZ stage or a PLY file."""
    if str(path).lower().endswith(".ply"):
        return newton.Gaussian.create_from_ply(path)

    from pxr import Usd  # noqa: PLC0415

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"Could not open Gaussian asset '{path}'")
    for prim in stage.Traverse():
        if str(prim.GetTypeName()).lower() == "particlefield3dgaussiansplat":
            return newton.Gaussian.create_from_usd(prim)
    raise ValueError(f"No ParticleField3DGaussianSplat prim found in '{path}'")


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.args = args
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.use_coupled = getattr(args, "solver", "coupled") == "coupled"

        gaussian = load_gaussian_asset(args.asset)
        gaussian_pos = np.asarray(gaussian.positions, dtype=np.float64)

        # Place the asset in the world: centered horizontally, standing on the ground.
        bbox_lower = gaussian_pos.min(axis=0)
        bbox_upper = gaussian_pos.max(axis=0)
        bbox_size = (bbox_upper - bbox_lower).max()
        cell_size = bbox_size / args.cage_resolution
        center_xy = 0.5 * (bbox_lower + bbox_upper)
        gaussian_pos = gaussian_pos - np.array([center_xy[0], center_xy[1], bbox_lower[2] - cell_size])
        self.asset_height = float(bbox_upper[2] - bbox_lower[2])

        # Voxel cage sized to the asset's longest axis, only where Gaussians are dense.
        vertices, tets, cell_of_point, keep_mask = build_tet_cage(gaussian_pos, cell_size, args.min_cell_points)

        # Keep only the Gaussians that fall inside the occupied cells.
        gaussian_pos = gaussian_pos[keep_mask]
        gaussian = newton.Gaussian(
            positions=gaussian_pos,
            rotations=gaussian.rotations[keep_mask],
            scales=gaussian.scales[keep_mask],
            opacities=gaussian.opacities[keep_mask],
            sh_coeffs=gaussian.sh_coeffs[keep_mask],
            sh_degree=gaussian.sh_degree,
            min_response=gaussian.min_response,
            sorting_mode=gaussian.sorting_mode,
        )
        tet_id, weights = embed_points_in_cage(gaussian_pos, vertices, tets, cell_of_point)

        # Rest-shape inverse used to extract the live rotation per tet.
        v = vertices[tets]
        rest_edges = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], axis=-1)
        rest_inverse = np.linalg.inv(rest_edges)

        self.gaussian_twin = GaussianTwin(gaussian, tet_id, weights, rest_inverse)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 2.0e4
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_ground_plane(color=(0.4, 0.4, 0.4))

        rigid_body_start = builder.body_count
        self._emit_rigid_bodies(builder)
        rigid_body_end = builder.body_count

        # Soft-body cage.  Particles are free and simulated by VBD; there is no
        # underlying body, which is why the Gaussian twin carries the visible field.
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(v) for v in vertices],
            indices=tets.flatten().tolist(),
            density=args.density,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            add_surface_mesh_edges=True,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=cell_size * 0.45,
            validate_mesh=True,
            label="gaussian_twin_cage",
        )

        # The render-only Gaussian twin: attached to the static world with an
        # identity xform because the skinning kernel writes world-space centers.
        builder.add_shape_gaussian(
            body=-1,
            gaussian=gaussian,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            scale=(1.0, 1.0, 1.0),
            collision_proxy=None,
            color=(1.0, 1.0, 1.0),
            label="bluehair_ragdoll_gaussians",
        )

        builder.color()
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_mu = 0.5

        vbd_kwargs = {
            "iterations": 10,
            "rigid_compliant_alm": True,
            "friction_epsilon": 0.01,
            "particle_enable_self_contact": True,
            "particle_self_contact_radius": cell_size * 0.35,
            "particle_self_contact_margin": cell_size * 0.7,
        }

        if self.use_coupled:
            rigid_body_indices = wp.array(list(range(rigid_body_start, rigid_body_end)), dtype=int)
            self.solver = SolverCoupledProxy(
                model=self.model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name="mujoco",
                        solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=200),
                        bodies=[int(i) for i in rigid_body_indices.numpy()],
                        joints=list(range(self.model.joint_count)),
                    ),
                    SolverCoupledProxy.Entry(
                        name="vbd",
                        solver=lambda v: SolverVBD(model=v, **vbd_kwargs),
                        bodies=[],
                        particles=list(range(self.model.particle_count)),
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="mujoco",
                            destination="vbd",
                            bodies=[int(i) for i in rigid_body_indices.numpy()],
                            mass_scale=args.mass_scale,
                            mode=args.coupling_mode,
                            collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                                model, self.args
                            ),
                            collide_interval=1,
                        )
                    ],
                    iterations=args.proxy_iterations,
                ),
            )
        else:
            self.solver = SolverVBD(model=self.model, **vbd_kwargs)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)
        if hasattr(self.viewer, "show_gaussians"):
            # Show the Gaussian twin and hide the tet cage's collision surface: the
            # visible appearance is meant to come from the splats only.
            self.viewer.show_gaussians = True
            self.viewer.show_triangles = args.show_cage
            self.viewer.gaussians_max_points = max(self.viewer.gaussians_max_points, gaussian.count)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.2, -1.8, 1.0), -15.0, 124.0)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.capture()

    def capture(self):
        with wp.ScopedDevice(self.model.device):
            with wp.ScopedCapture() as capture:
                self.simulate()
        self.graph = capture.graph

    def simulate(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            with wp.ScopedDevice(self.model.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.gaussian_twin.update(self.model, self.state_0)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def test_final(self):
        # The rigid bodies and the soft toy should not have tumbled through the ground.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all rigid bodies are above the ground",
            lambda q, qd: q[2] > -0.05,
        )
        particle_q = self.state_0.particle_q.numpy()
        assert np.min(particle_q, axis=0)[2] > -0.1, "Soft-body cage fell through the ground"
        assert np.max(particle_q, axis=0)[2] < 2.5, "Soft-body cage exploded upward"
        gaussian_q = self.gaussian_twin.gaussian.warp_data.transforms.numpy()[:, :3]
        assert np.isfinite(gaussian_q).all(), "Gaussian field contains non-finite positions"

    def _emit_rigid_bodies(self, builder: newton.ModelBuilder):
        """Add a swinging paddle and a dropping box that knock the toy over."""
        # Paddle anchored off to the side, initially lifted so gravity swings it down.
        anchor = wp.vec3(0.5, 0.0, self.asset_height + 0.1)
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.45, hy=0.03, hz=0.05, color=(0.8, 0.5, 0.2))
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=anchor, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.45, 0.0, 0.0), q=wp.quat_identity()),
            target_kd=2.0,
        )
        builder.add_articulation([joint], label="paddle")
        # Raise the arm (single revolute coordinate) so gravity swings it into the toy.
        builder.joint_q[-1] = 1.2

        # A box dropped onto the ragdoll from above.
        box = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.05, 0.0, self.asset_height + 0.45), q=wp.quat_identity()),
            mass=2.0,
        )
        builder.add_shape_box(box, hx=0.08, hy=0.08, hz=0.08, color=(0.3, 0.6, 0.8))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument(
            "--asset",
            type=str,
            default=DEFAULT_ASSET,
            help="Path to the Gaussian splat USDZ/PLY asset.",
        )
        parser.add_argument(
            "--cage-resolution",
            type=int,
            default=20,
            help="Number of voxels along the asset's longest axis.",
        )
        parser.add_argument(
            "--min-cell-points",
            type=int,
            default=8,
            help="Minimum number of Gaussian centers required to keep a cage cell.",
        )
        parser.add_argument(
            "--density",
            type=float,
            default=300.0,
            help="Soft-body cage density in kg/m^3.",
        )
        parser.add_argument(
            "--k-mu",
            type=float,
            default=1.5e5,
            help="Soft-body Lame parameter mu (Pa).",
        )
        parser.add_argument(
            "--k-lambda",
            type=float,
            default=1.5e5,
            help="Soft-body Lame parameter lambda (Pa).",
        )
        parser.add_argument(
            "--k-damp",
            type=float,
            default=1.0e2,
            help="Soft-body viscous damping (Pa*s).",
        )
        parser.add_argument(
            "--solver",
            "-s",
            type=str,
            choices=["coupled", "vbd"],
            default="coupled",
            help="'coupled' for rigid+VBD coupling, 'vbd' for pure-VBD baseline.",
        )
        parser.add_argument(
            "--mass-scale",
            "-pmr",
            type=float,
            default=1.0,
            help="Proxy mass scale used by VBD when coupled to MuJoCo.",
        )
        parser.add_argument(
            "--coupling-mode",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
            help="Proxy coupling sync mode.",
        )
        parser.add_argument(
            "--proxy-iterations",
            type=int,
            default=1,
            help="Number of proxy relaxation passes per substep.",
        )
        parser.add_argument(
            "--show-cage",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Show the VBD tetrahedral cage surface (default: off so only the Gaussian twin is visible).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
