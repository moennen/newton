# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid-VBD Coupled Solver with a Gaussian Deformable Twin
#
# A plush toy captured as a 3D Gaussian splat is simulated as a volumetric
# soft body: the asset ships its own tetrahedral simulation mesh plus a
# baked skinning binding of every Gaussian to that mesh, so the tet mesh is
# simulated by VBD while the Gaussian field stays render-only and is skinned
# to it every frame (a "deformable twin").
#
# The asset is a single USD file holding
#
#   * ``SimulationMesh``, a ``UsdGeom.TetMesh`` imported by ``add_usd`` as a
#     VBD soft body,
#   * ``Gaussians``, a ``ParticleField3DGaussianSplat`` imported as a
#     Gaussian shape, carrying the ``newton:deformableSkin:*`` attributes:
#     a linear-blend skinning binding that names, per Gaussian, the
#     simulation mesh vertices that drive it and their blend weights, and
#   * ``VisualMesh``, a render surface sharing the simulation mesh vertices.
#     The example ignores it on import: it would come in as a static shape,
#     and --show-tetmesh already draws that surface from the live mesh.
#
# Nothing is voxelized or re-fitted at runtime: the example only reads the
# authored binding. Gaussian centers follow the blended nodal displacements
# and Gaussian orientations follow the rotation of the deformation gradient
# blended with the same weights. The Gaussian field carries no mass and no
# collision geometry; the tet mesh owns all physical state.
#
# Two scenes exercise rigid/soft coupling through ``SolverCoupledProxy``,
# with the rigid bodies driven by MuJoCo and the toy by VBD:
#
#   * ``grasp`` (default): a parallel-jaw gripper pinches the toy around
#     the waist and lifts it off the ground.
#   * ``sway``: the same grasp, then the carriage swings the toy back and
#     forth so the skinned field lags and wobbles.
#
# Only the 'gl' and 'viser' viewers draw Gaussian assets; other backends
# fall back to rendering the simulation mesh surface. Pass --show-tetmesh
# to see that surface alongside the splats.
#
# Command: python -m newton.examples mujoco_vbd_gaussian_twin
#          python -m newton.examples mujoco_vbd_gaussian_twin --scene sway
#          python -m newton.examples mujoco_vbd_gaussian_twin --show-tetmesh
#
###########################################################################

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.viewer import ViewerBase

# Folder and file of the packaged asset in the newton-assets repository. Set
# ``--asset`` or the NEWTON_GAUSSIAN_TWIN_ASSET environment variable to use a
# local package instead of the downloaded one.
ASSET_FOLDER = "gaussian_splat_toys"
ASSET_FILE = "baked.BluehairRagdoll_package.usda"

# Namespace of the skinning attributes authored on the asset.
SKIN = "newton:deformableSkin"

# Prims the physics import must not pick up: the visual surface is skinned for
# rendering only and would otherwise be imported as a static shape.
IGNORE_PATHS = [".*VisualMesh"]


def resolve_asset(path: str | None) -> str:
    """Return the asset path to load, downloading the packaged asset if needed."""
    return (
        path
        or os.environ.get("NEWTON_GAUSSIAN_TWIN_ASSET")
        or str(newton.utils.download_asset(ASSET_FOLDER) / ASSET_FILE)
    )


@dataclass(frozen=True)
class SkinBinding:
    """Linear-blend skinning of a Gaussian field to a simulation mesh, baked into the asset.

    Attributes:
        influence_indices: Simulation mesh vertices driving each Gaussian, shape ``[num_gaussians, num_influences]``.
        influence_weights: Blend weight of each influence, shape ``[num_gaussians, num_influences]``.
        points: Rest positions of the simulation mesh [m], shape ``[num_vertices, 3]``.
        tet_indices: Vertices of each tetrahedron, shape ``[num_tets, 4]``.
    """

    influence_indices: np.ndarray
    influence_weights: np.ndarray
    points: np.ndarray
    tet_indices: np.ndarray


def read_skin_binding(asset_path: str) -> SkinBinding:
    """Read the baked Gaussian skinning binding from *asset_path*.

    Raises:
        ValueError: If the asset does not carry a usable ``newton:deformableSkin`` binding.
    """
    from pxr import Usd  # noqa: PLC0415

    stage = Usd.Stage.Open(asset_path)
    if stage is None:
        raise ValueError(f"unable to open USD stage '{asset_path}'")
    gaussian_prim = next((p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"), None)
    tet_prim = next((p for p in stage.Traverse() if p.GetTypeName() == "TetMesh"), None)
    if gaussian_prim is None or tet_prim is None:
        raise ValueError(f"'{asset_path}' must hold both a ParticleField3DGaussianSplat and a UsdGeom.TetMesh prim")

    def read(prim, name):
        attr = prim.GetAttribute(name)
        if not attr or attr.Get() is None:
            raise ValueError(f"{prim.GetPath()} is missing the attribute '{name}'")
        return attr.Get()

    count = int(read(gaussian_prim, f"{SKIN}:pointCount"))
    influences = int(read(gaussian_prim, f"{SKIN}:influenceSize"))
    if count <= 0 or influences <= 0:
        raise ValueError(f"unusable skinning binding: pointCount={count}, influenceSize={influences}")

    points = np.asarray(read(tet_prim, "points"), dtype=np.float32)
    binding = SkinBinding(
        influence_indices=np.asarray(read(gaussian_prim, f"{SKIN}:influenceIndices"), dtype=np.int32).reshape(
            -1, influences
        ),
        influence_weights=np.asarray(read(gaussian_prim, f"{SKIN}:influenceWeights"), dtype=np.float32).reshape(
            -1, influences
        ),
        points=points,
        tet_indices=np.asarray(read(tet_prim, "tetVertexIndices"), dtype=np.int32).reshape(-1, 4),
    )
    if len(binding.influence_indices) != count or len(binding.influence_weights) != count:
        raise ValueError(f"skinning binding does not match the authored pointCount of {count}")
    if binding.influence_indices.min() < 0 or binding.influence_indices.max() >= len(points):
        raise ValueError("skinning influences do not index the simulation mesh vertices")
    return binding


def tet_rest_bases(binding: SkinBinding) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the per-tetrahedron quantities the deformation gradient pass needs.

    Returns the inverse rest edge basis per tetrahedron [1/m], the rest volume
    each tetrahedron contributes to its vertices [m^3], and the reciprocal of the
    volume accumulated per vertex [1/m^3]. Degenerate tetrahedra (the packaged
    meshes contain a few slivers) get a zero basis and zero volume, so they drop
    out of the vertex average instead of polluting it.
    """
    corners = binding.points[binding.tet_indices]
    edges = np.stack([corners[:, k] - corners[:, 0] for k in (1, 2, 3)], axis=-1)
    determinant = np.linalg.det(edges)
    usable = np.abs(determinant) > 1.0e-16
    basis_inv = np.zeros_like(edges)
    basis_inv[usable] = np.linalg.inv(edges[usable])
    volume = np.where(usable, np.abs(determinant) / 6.0, 0.0)

    accumulated = np.zeros(len(binding.points))
    np.add.at(accumulated, binding.tet_indices.ravel(), np.repeat(volume, 4))
    inv_accumulated = np.where(accumulated > 0.0, 1.0 / np.maximum(accumulated, 1.0e-30), 0.0)
    return (
        basis_inv.astype(np.float32),
        volume.astype(np.float32),
        inv_accumulated.astype(np.float32),
    )


@wp.kernel
def accumulate_vertex_gradients(
    particle_q: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    rest_basis_inv: wp.array[wp.mat33],
    tet_volume: wp.array[float],
    vertex_gradient: wp.array[wp.mat33],
):
    """Scatter each tetrahedron's volume-weighted deformation gradient onto its vertices."""
    t = wp.tid()
    i0 = tet_indices[t, 0]
    x0 = particle_q[i0]

    # Deformation gradient F = live_edges * rest_edges^-1, constant over the element.
    F = (
        wp.matrix_from_cols(
            particle_q[tet_indices[t, 1]] - x0,
            particle_q[tet_indices[t, 2]] - x0,
            particle_q[tet_indices[t, 3]] - x0,
        )
        * rest_basis_inv[t]
    )
    contribution = F * tet_volume[t]
    for k in range(4):
        wp.atomic_add(vertex_gradient, tet_indices[t, k], contribution)


@wp.kernel
def skin_gaussians_to_mesh(
    particle_q: wp.array[wp.vec3],
    rest_particle_q: wp.array[wp.vec3],
    vertex_gradient: wp.array[wp.mat33],
    vertex_inv_volume: wp.array[float],
    influence_indices: wp.array2d[wp.int32],
    influence_weights: wp.array2d[float],
    rest_position: wp.array[wp.vec3],
    rest_rotation: wp.array[wp.quat],
    transforms_out: wp.array[wp.transform],
):
    """Update Gaussian splat centers/rotations from the live VBD simulation mesh.

    Centers follow the weighted blend of the nodal displacements of the bound
    vertices, which reproduces the authored center exactly at rest even where
    the baked weights extrapolate outside the mesh. Orientations follow the
    rotation part of the deformation gradient blended with the same weights,
    which stays smooth across element boundaries.
    """
    i = wp.tid()
    center = rest_position[i]
    gradient = wp.mat33(0.0)
    for k in range(influence_indices.shape[1]):
        v = influence_indices[i, k]
        w = influence_weights[i, k]
        center += (particle_q[v] - rest_particle_q[v]) * w
        gradient += vertex_gradient[v] * (w * vertex_inv_volume[v])

    # Rotation part of the blended gradient, extracted by Gram-Schmidt
    # orthonormalization of its columns. Stretch is intentionally dropped: only
    # Gaussian centers and orientations are skinned, covariance stays at its
    # bind-pose value.
    c0 = wp.vec3(gradient[0, 0], gradient[1, 0], gradient[2, 0])
    c1 = wp.vec3(gradient[0, 1], gradient[1, 1], gradient[2, 1])
    if wp.length(c0) < 1.0e-9 or wp.length(wp.cross(c0, c1)) < 1.0e-12:
        # Vertices without a usable gradient (no incident tetrahedron, or a
        # collapsed neighborhood) keep the bind-pose orientation.
        transforms_out[i] = wp.transform(center, rest_rotation[i])
        return

    c0 = wp.normalize(c0)
    c1 = wp.normalize(c1 - c0 * wp.dot(c0, c1))
    c2 = wp.cross(c0, c1)

    q_rot = wp.quat_from_matrix(wp.matrix_from_cols(c0, c1, c2))
    transforms_out[i] = wp.transform(center, wp.normalize(q_rot * rest_rotation[i]))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of ``xyzw`` quaternions, broadcasting over leading axes."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


@wp.kernel
def set_joint_target(joint_target_q: wp.array[float], index: int, value: float):
    joint_target_q[index] = value


class GaussianTwin:
    """Render-only Gaussian field skinned to a simulated tetrahedral mesh."""

    def __init__(
        self,
        binding: SkinBinding,
        gaussian: newton.Gaussian,
        xform: wp.transform,
        rest_particle_q: wp.array,
    ):
        device = rest_particle_q.device
        self.transforms = gaussian.warp_data.transforms
        self.rest_particle_q = rest_particle_q

        # The importer put the asset's placement on the Gaussian shape transform;
        # the example zeroes it and folds it into the bind pose instead, so the
        # kernel can write world-space transforms directly.
        rest = self.transforms.numpy()
        rotation = np.array(wp.quat_to_matrix(xform.q), dtype=np.float32).reshape(3, 3)
        rest_position = rest[:, 0:3] @ rotation.T + np.array(xform.p, dtype=np.float32)
        rest_rotation = quat_mul(np.array(xform.q, dtype=np.float32), rest[:, 3:7])

        basis_inv, tet_volume, vertex_inv_volume = tet_rest_bases(binding)
        self.gradient_inputs = [
            wp.array(binding.tet_indices, dtype=wp.int32, device=device),
            wp.array(basis_inv, dtype=wp.mat33, device=device),
            wp.array(tet_volume, dtype=float, device=device),
        ]
        self.num_tets = len(binding.tet_indices)
        self.vertex_gradient = wp.zeros(len(binding.points), dtype=wp.mat33, device=device)
        self.skin_inputs = [
            wp.array(vertex_inv_volume, dtype=float, device=device),
            wp.array(binding.influence_indices, dtype=wp.int32, device=device),
            wp.array(binding.influence_weights, dtype=float, device=device),
            wp.array(rest_position, dtype=wp.vec3, device=device),
            wp.array(rest_rotation, dtype=wp.quat, device=device),
        ]

    def update(self, state: newton.State) -> None:
        """Skin the Gaussian field to the current particle positions."""
        device = self.transforms.device
        self.vertex_gradient.zero_()
        wp.launch(
            accumulate_vertex_gradients,
            dim=self.num_tets,
            inputs=[state.particle_q, *self.gradient_inputs],
            outputs=[self.vertex_gradient],
            device=device,
        )
        wp.launch(
            skin_gaussians_to_mesh,
            dim=len(self.transforms),
            inputs=[state.particle_q, self.rest_particle_q, self.vertex_gradient, *self.skin_inputs],
            outputs=[self.transforms],
            device=device,
        )


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.args = args
        self.viewer = viewer
        self.scene = args.scene
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.asset = resolve_asset(args.asset)
        binding = read_skin_binding(self.asset)
        # Lay the toy on its back on the ground plane: a plush ragdoll standing on two thin
        # legs topples within a tenth of a second, long before jaws driven at a frequency the
        # substeps can resolve would reach it, so the scene picks it up off the ground. The
        # rotation turns the asset's long axis onto +x, its width onto the closing direction
        # +y and its thickness onto +z, and the offset centers the toy over the origin.
        lay_down = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 1.0)), 2.0 * math.pi / 3.0)
        rest = binding.points @ np.array(wp.quat_to_matrix(lay_down), dtype=np.float32).reshape(3, 3).T

        # Packaged toys differ in size and shape, so the rig and the motion schedule are sized
        # from the toy's own dimensions and from the waist, the cross-section the jaws close
        # on. The waist is what gets centered over the origin rather than the whole bounding
        # box: a captured toy is not symmetric, and centering the box leaves the waist off to
        # one side, where one jaw squeezes while the other closes on air.
        self.toy_length, self.toy_width, self.toy_thickness = (float(extent) for extent in np.ptp(rest, axis=0))
        middle = 0.5 * float(rest[:, 0].min() + rest[:, 0].max())
        waist = np.abs(rest[:, 0] - middle) < 0.15 * self.toy_length
        waist_y = rest[waist, 1]
        self.waist_half_width = 0.5 * float(waist_y.max() - waist_y.min())
        offset = np.array(
            [-middle, -0.5 * float(waist_y.min() + waist_y.max()), -float(rest[:, 2].min())], dtype=np.float32
        )
        toy_xform = wp.transform(wp.vec3(*offset.tolist()), lay_down)
        rest = rest + offset
        corners = binding.points[binding.tet_indices]
        edge_length = float(
            np.mean([np.linalg.norm(corners[:, a] - corners[:, b], axis=1) for a, b in ((0, 1), (0, 2), (0, 3))])
        )
        self.particle_radius = args.particle_radius if args.particle_radius > 0.0 else 0.4 * edge_length

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder)

        # The asset authors no physics material, so the toy's material comes from the
        # builder defaults that the TetMesh import picks up.
        poisson = args.poissons_ratio
        builder.default_tet_density = args.density
        builder.default_tet_k_mu = args.youngs_modulus / (2.0 * (1.0 + poisson))
        builder.default_tet_k_lambda = args.youngs_modulus * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        # Viscous damping is a stress per unit strain rate, so the rate it imposes on an
        # element grows as the elements get smaller. Fixing the damping ratio against the
        # element's own elastic response instead keeps the toy equally damped at any mesh
        # resolution and toy size, where a fixed [Pa*s] value would diverge on fine meshes.
        builder.default_tet_k_damp = (
            2.0 * args.damping_ratio * edge_length * math.sqrt(args.youngs_modulus * args.density)
        )
        builder.default_particle_radius = self.particle_radius

        results = builder.add_usd(self.asset, xform=toy_xform, ignore_paths=IGNORE_PATHS)
        self.toy_particles = list(range(builder.particle_count))
        # The baked binding indexes the authored mesh, so it only describes the imported
        # body if the import did not move the vertices around. A stage that declares no
        # up axis, for instance, is taken as Y-up and comes in rotated, which would place
        # the toy on its side and skin every Gaussian to the wrong part of it.
        imported = np.asarray(builder.particle_q, dtype=np.float32)
        if imported.shape != rest.shape or not np.allclose(imported, rest, atol=1.0e-5):
            raise ValueError(
                f"'{self.asset}' imported {len(imported)} simulation mesh vertices that do not match the "
                f"{len(rest)} vertices its skinning binding refers to; check that the stage declares "
                "'upAxis = Z' and 'metersPerUnit = 1'"
            )
        gaussian_shape = next(
            shape
            for shape in results["path_shape_map"].values()
            if builder.shape_type[shape] == newton.GeoType.GAUSSIAN
        )
        builder.shape_transform[gaussian_shape] = wp.transform_identity()

        builder.add_ground_plane(color=(0.4, 0.4, 0.4))
        rigid_bodies, rigid_joints = self._emit_gripper(builder)

        builder.color()
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        # Contacts act on single particles, so their stiffness has to follow the particle
        # mass: a packaged toy weighs a few grams, and kilogram-scale contact gains put the
        # contact frequency orders of magnitude above the substep rate, which throws the
        # particles off on the very first substep. Deriving the gain from the mean particle
        # mass and a contact frequency keeps the response resolvable at any toy scale.
        particle_mass = float(np.mean(1.0 / self.model.particle_inv_mass.numpy()))
        contact_omega = 2.0 * math.pi * args.contact_frequency
        self.model.soft_contact_ke = particle_mass * contact_omega * contact_omega
        self.model.soft_contact_kd = 1.0e-4
        # Keep tangential contact at the same baseline gain as normal contact. In this scene,
        # changing this gain made no measurable difference to retention; VBD iteration count
        # is the parameter that determines whether the pinched toy is lifted.
        self.model.soft_contact_kf = self.model.soft_contact_ke
        self.model.soft_contact_mu = 2.0

        vbd_kwargs = {
            "iterations": args.vbd_iterations,
            "rigid_compliant_alm": True,
            "friction_epsilon": 0.01,
            "particle_enable_self_contact": False,
            "rigid_body_particle_contact_buffer_size": 2048,
        }
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mujoco",
                    # All rigid contacts in both scenes are against the toy, and those are
                    # handled by the VBD entry through the proxy.
                    solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, disable_contacts=True),
                    bodies=rigid_bodies,
                    joints=rigid_joints,
                ),
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, **vbd_kwargs),
                    particles=self.toy_particles,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mujoco",
                        destination="vbd",
                        bodies=rigid_bodies,
                        # The actuated joints stay enabled in the VBD view so its proxy bodies
                        # remain rigidly connected to the driven articulation.
                        joints=rigid_joints,
                        mass_scale=args.mass_scale,
                        mode=args.coupling_mode,
                        collision_pipeline=lambda model: newton.examples.create_collision_pipeline(model, self.args),
                        collide_interval=1,
                    )
                ],
                iterations=args.proxy_iterations,
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.twin = GaussianTwin(
            binding,
            self.model.shape_source[gaussian_shape],
            toy_xform,
            wp.clone(self.state_0.particle_q),
        )
        self.twin.update(self.state_0)
        self.initial_tet_center = np.mean(self.state_0.particle_q.numpy(), axis=0)
        self.initial_gaussian_center = np.mean(self.twin.transforms.numpy()[:, 0:3], axis=0)

        target_q_start = self.model.joint_target_q_start.numpy()
        self.lift_target = int(target_q_start[self.lift_joint])
        self.sway_target = int(target_q_start[self.sway_joint])
        self.finger_targets = (int(target_q_start[self.left_joint]), int(target_q_start[self.right_joint]))
        self.max_toy_swing = 0.0

        newton.examples.configure_coupled_view(self, args)
        # Only some backends draw Gaussian assets; ``log_gaussian`` is a no-op in the
        # base viewer. Without it the toy would be invisible, so fall back to the
        # simulation mesh surface and say why.
        splats_supported = type(self.viewer).log_gaussian is not ViewerBase.log_gaussian
        if not splats_supported and not args.quiet:
            print(
                f"{type(self.viewer).__name__} cannot render Gaussian splats; showing the simulation "
                "mesh instead. Run with '--viewer gl' or '--viewer viser' to see the Gaussian twin."
            )
        if hasattr(self.viewer, "show_gaussians"):
            # The visible appearance is meant to come from the splats alone, so the tet
            # mesh surface stays hidden unless it is asked for or nothing else would show.
            self.viewer.show_gaussians = splats_supported
            self.viewer.show_triangles = args.show_tetmesh or not splats_supported
            self.viewer.gaussians_max_points = max(self.viewer.gaussians_max_points, len(self.twin.transforms))
        if hasattr(self.viewer, "set_camera"):
            length = self.toy_length
            self.viewer.set_camera(wp.vec3(1.4 * length, -1.7 * length, 0.9 * length), -18.0, 130.0)

        self.capture()

    def _emit_gripper(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
        """Add a parallel-jaw gripper on a lift/sway carriage that pinches the toy's waist.

        The carriage rides the world on a vertical and a horizontal prismatic joint so the
        same rig can lift the toy and, in the ``sway`` scene, swing it back and forth. All
        dimensions scale with the toy, and the drive gains and effort limits are derived
        from the mass they actually carry, so any packaged asset is gripped the same way.
        A packaged toy weighs a few grams, and gains stiff enough for a kilogram would put
        the drives far above the substep rate and blow the articulation up.
        """
        cfg = newton.ModelBuilder.ShapeConfig(density=800.0, ke=8.0e4, kd=1.0e-4, kf=1.0e3, mu=1.0)
        length = self.toy_length
        # The jaws straddle the waist, clear of it when open and squeezing into it when
        # closed, so both the gap and the travel come from the waist's own half width.
        grip_z = self.args.grip_height * self.toy_thickness
        open_y = self.args.finger_open * self.waist_half_width
        self.finger_close = (self.args.finger_open - self.args.pinch_depth) * self.waist_half_width
        back_x = -0.7 * length
        hub = 0.06 * length
        finger_color = wp.vec3(0.88, 0.48, 0.22)

        carriage_xform = wp.transform(wp.vec3(back_x - 1.5 * hub, 0.0, grip_z), wp.quat_identity())
        carriage = builder.add_link(xform=carriage_xform, label="carriage")
        builder.add_shape_box(carriage, hx=hub, hy=hub, hz=hub, cfg=cfg, color=wp.vec3(0.3, 0.32, 0.36))
        palm = builder.add_link(xform=wp.transform(wp.vec3(back_x, 0.0, grip_z), wp.quat_identity()), label="palm")
        builder.add_shape_box(palm, hx=hub, hy=open_y, hz=hub, cfg=cfg, color=wp.vec3(0.22, 0.28, 0.34))
        finger_hx = 0.18 * length
        finger_hy = 0.02 * length
        finger_hz = 0.5 * self.toy_thickness
        left = builder.add_link(xform=wp.transform(wp.vec3(0.0, open_y, grip_z), wp.quat_identity()), label="left")
        builder.add_shape_box(left, hx=finger_hx, hy=finger_hy, hz=finger_hz, cfg=cfg, color=finger_color)
        right = builder.add_link(xform=wp.transform(wp.vec3(0.0, -open_y, grip_z), wp.quat_identity()), label="right")
        builder.add_shape_box(right, hx=finger_hx, hy=finger_hy, hz=finger_hz, cfg=cfg, color=finger_color)

        # Critically damped position drives at args.drive_frequency for the mass the rig
        # carries: the toy plus the gripper itself. Effort limits leave ample headroom over
        # that weight so the jaws can pinch and the carriage can accelerate.
        payload = sum(builder.particle_mass) + sum(builder.body_mass)
        omega = 2.0 * math.pi * self.args.drive_frequency
        target_ke = payload * omega * omega
        target_kd = 2.0 * payload * omega
        effort_limit = self.args.effort_scale * payload * 9.81

        self.lift_joint = builder.add_joint_prismatic(
            parent=-1,
            child=carriage,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=carriage_xform,
            target_ke=target_ke,
            target_kd=target_kd,
            effort_limit=effort_limit,
            label="lift",
        )
        self.sway_joint = builder.add_joint_prismatic(
            parent=carriage,
            child=palm,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(1.5 * hub, 0.0, 0.0), wp.quat_identity()),
            target_ke=target_ke,
            target_kd=target_kd,
            effort_limit=effort_limit,
            label="sway",
        )
        finger_joints = []
        for name, body, sign in (("left", left, 1.0), ("right", right, -1.0)):
            finger_joints.append(
                builder.add_joint_prismatic(
                    parent=palm,
                    child=body,
                    axis=wp.vec3(0.0, -sign, 0.0),
                    parent_xform=wp.transform(wp.vec3(-back_x, sign * open_y, 0.0), wp.quat_identity()),
                    target_ke=target_ke,
                    target_kd=target_kd,
                    limit_lower=0.0,
                    limit_upper=open_y,
                    limit_ke=target_ke,
                    limit_kd=target_kd,
                    effort_limit=effort_limit,
                    label=f"{name}_finger",
                )
            )
        self.left_joint, self.right_joint = finger_joints
        joints = [self.lift_joint, self.sway_joint, *finger_joints]
        builder.add_articulation(joints, label="gripper")
        return [carriage, palm, left, right], joints

    def _set_target(self, index: int, value: float) -> None:
        wp.launch(
            set_joint_target,
            dim=1,
            inputs=[self.control.joint_target_q, index, value],
            device=self.model.device,
        )

    def _update_targets(self) -> None:
        """Close the jaws, lift the toy, then sway the carriage in the 'sway' scene."""
        args = self.args
        length = self.toy_length
        lift_start = args.close_time
        close = self.finger_close * min(1.0, self.sim_time / args.close_time)
        lift = args.lift_height * length * min(1.0, max(0.0, self.sim_time - lift_start) / args.lift_time)
        for index in self.finger_targets:
            self._set_target(index, close)
        self._set_target(self.lift_target, lift)
        if self.scene == "sway":
            # Swing the lifted toy horizontally so the skinned field shows inertial lag.
            swing = max(0.0, self.sim_time - lift_start - args.lift_time)
            # A plain sine starts at full speed, and that jerk shakes the toy out of the
            # jaws, so the amplitude fades in over the first period.
            amplitude = args.sway_distance * length * min(1.0, swing / args.sway_period)
            self._set_target(self.sway_target, amplitude * math.sin(2.0 * math.pi * swing / args.sway_period))

    def capture(self):
        self.graph = None
        with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
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
        self._update_targets()
        if self.graph is not None:
            with wp.ScopedDevice(self.model.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.twin.update(self.state_0)
        swing = abs(float(np.mean(self.state_0.particle_q.numpy()[:, 0])) - self.initial_tet_center[0])
        self.max_toy_swing = max(self.max_toy_swing, swing)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def test_final(self):
        length = self.toy_length
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all rigid bodies are above the ground",
            lambda q, qd: q[2] > -0.15 * length,
        )
        particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all(), "Simulation mesh contains non-finite positions"
        assert particle_q[:, 2].min() > -0.3 * length, "Simulation mesh fell through the ground"
        assert particle_q[:, 2].max() < 8.0 * length, "Simulation mesh exploded upward"

        tet_shift = np.mean(particle_q, axis=0) - self.initial_tet_center
        # Short runs (the CI smoke test uses two frames) stop before the gripper has
        # moved, so only check the outcome once the motion schedule has played out.
        lift_done = self.args.close_time + self.args.lift_time
        if self.sim_time > lift_done:
            assert tet_shift[2] > 0.15 * length, f"Gripper did not lift the toy: dz={tet_shift[2]:.4f}"
        if self.scene == "sway" and self.sim_time > lift_done + 0.5 * self.args.sway_period:
            assert self.max_toy_swing > 0.06 * length, (
                f"Carriage did not swing the toy: max dx={self.max_toy_swing:.4f}"
            )

        gaussian_q = self.twin.transforms.numpy()[:, 0:3]
        assert np.isfinite(gaussian_q).all(), "Gaussian field contains non-finite positions"
        # The skinned field must follow the simulation mesh it is bound to. The two
        # centers are not identical (the field extends past the mesh where the baked
        # weights extrapolate), so compare how far each has travelled.
        drift = np.linalg.norm(np.mean(gaussian_q, axis=0) - self.initial_gaussian_center - tet_shift)
        assert drift < 0.15 * length, f"Gaussian field did not follow the simulation mesh: drift={drift:.4f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.set_defaults(num_frames=180)
        parser.add_argument(
            "--scene",
            type=str,
            choices=["grasp", "sway"],
            default="grasp",
            help="'grasp' pinches and lifts the toy, 'sway' also swings it once lifted.",
        )
        parser.add_argument(
            "--asset",
            type=str,
            default=None,
            help="USD package holding the Gaussian field, its TetMesh, and the baked skinning binding. "
            f"Defaults to '{ASSET_FILE}' downloaded from newton-assets.",
        )
        parser.add_argument("--substeps", type=int, default=32, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=60, help="VBD iterations per substep.")
        parser.add_argument("--density", type=float, default=300.0, help="Toy density [kg/m^3].")
        parser.add_argument("--youngs-modulus", type=float, default=3.5e4, help="Toy Young's modulus [Pa].")
        parser.add_argument("--poissons-ratio", type=float, default=0.35, help="Toy Poisson's ratio.")
        parser.add_argument(
            "--damping-ratio",
            type=float,
            default=0.1,
            help="Toy damping, as a fraction of the critical damping of one element. Converted to a "
            "viscous damping [Pa*s] using the mesh resolution and the material.",
        )
        parser.add_argument(
            "--particle-radius",
            type=float,
            default=0.0,
            help="Simulation mesh contact radius [m]. Defaults to 0.4 of the mean tetrahedron edge length.",
        )
        parser.add_argument(
            "--grip-height", type=float, default=0.6, help="Height of the gripper jaws [toy thicknesses]."
        )
        parser.add_argument(
            "--drive-frequency",
            type=float,
            default=25.0,
            help="Undamped natural frequency of the gripper position drives [Hz]. The gains follow from "
            "this and the mass the rig carries, so it must stay well below the substep rate.",
        )
        parser.add_argument(
            "--contact-frequency",
            type=float,
            default=300.0,
            help="Undamped natural frequency of a particle contact [Hz]. Sets the soft contact gains "
            "from the mean particle mass, so it must also stay below the substep rate.",
        )
        parser.add_argument(
            "--effort-scale",
            type=float,
            default=5.0,
            help="Gripper effort limit as a multiple of the weight it carries. Bounds the pinch force, "
            "which only has to beat the toy's weight through friction.",
        )
        parser.add_argument(
            "--finger-open",
            type=float,
            default=1.8,
            help="Half gap between the open jaws [half widths of the toy's waist].",
        )
        parser.add_argument(
            "--pinch-depth",
            type=float,
            default=0.7,
            help="Half gap between the closed jaws [half widths of the toy's waist]. Below 1.0 the jaws "
            "squeeze into the toy, which is what friction needs to hold it.",
        )
        parser.add_argument("--close-time", type=float, default=0.6, help="Finger closing ramp duration [s].")
        parser.add_argument(
            "--lift-height", type=float, default=0.5, help="Height the gripper lifts the toy to [toy lengths]."
        )
        parser.add_argument("--lift-time", type=float, default=1.0, help="Lifting ramp duration [s].")
        parser.add_argument(
            "--sway-distance", type=float, default=0.37, help="Sway amplitude of the carriage [toy lengths]."
        )
        parser.add_argument("--sway-period", type=float, default=1.0, help="Sway period of the carriage [s].")
        parser.add_argument("--mass-scale", type=float, default=1.0, help="Proxy mass scale used by the VBD entry.")
        parser.add_argument(
            "--coupling-mode",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
            help="Proxy coupling sync mode.",
        )
        parser.add_argument(
            "--proxy-iterations", type=int, default=1, help="Number of proxy relaxation passes per substep."
        )
        parser.add_argument(
            "--show-tetmesh",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Show the VBD simulation mesh surface (default: off so only the Gaussian twin is visible).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
