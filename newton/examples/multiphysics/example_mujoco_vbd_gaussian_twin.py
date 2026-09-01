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
# authored binding. Gaussian centers follow the blended nodal displacements;
# optional orientations and scales follow the local deformation gradient. The
# Gaussian field carries no mass and no collision geometry; the tet mesh owns
# all physical state.
#
# Two scenes exercise rigid/soft coupling through ``SolverCoupledProxy``,
# with the rigid bodies driven by MuJoCo and the toy by VBD:
#
#   * ``grasp`` (default): a parallel-jaw gripper pinches the toy around
#     the waist and lifts it off the ground.
#   * ``sway``: the same grasp, then the carriage swings the toy back and
#     forth so the skinned field lags and wobbles.
#
# The RTX viewer ray traces the Gaussian assets, and the USD viewer records
# them as native ``ParticleField3DGaussianSplat`` prims. Other backends fall
# back to rendering the simulation mesh surface. Pass --show-tetmesh to see
# that surface alongside the splats.
#
# Command: python -m newton.examples mujoco_vbd_gaussian_twin --asset path/to/package.usda
#          python -m newton.examples mujoco_vbd_gaussian_twin --asset path/to/package.usda --scene sway
#          python -m newton.examples mujoco_vbd_gaussian_twin --asset path/to/package.usda --show-tetmesh
#          python -m newton.examples mujoco_vbd_gaussian_twin --asset path/to/package.usda --fast-simulation
#
###########################################################################

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.viewer import ViewerBase, ViewerViser

# The packaged asset is not distributed with Newton. Set ``--asset`` or
# ``NEWTON_GAUSSIAN_TWIN_ASSET`` to its USD package path.

# Namespace of the skinning attributes authored on the asset.
SKIN = "newton:deformableSkin"

# Prims the physics import must not pick up: the visual surface is skinned for
# rendering only and would otherwise be imported as a static shape.
IGNORE_PATHS = [".*VisualMesh"]


# This scene's default is deliberately conservative: it resolves the elastic
# wave travel time of every tetrahedron and runs enough VBD iterations for the
# gripper contact to be close to converged.  The balanced and real-time
# presets trade some temporal and nonlinear-solve accuracy for interactive
# cost. Their reduced drive bandwidth avoids the finger/contact limit cycle
# that the 25 Hz default exhibits with a coarser solve.
BALANCED_SIMULATION_SUBSTEPS = 8
BALANCED_SIMULATION_VBD_ITERATIONS = 45
BALANCED_SIMULATION_DRIVE_FREQUENCY = 12.5
FAST_SIMULATION_SUBSTEPS = 4
FAST_SIMULATION_VBD_ITERATIONS = 30
FAST_SIMULATION_DRIVE_FREQUENCY = 7.5


def resolve_asset(path: str | None) -> str:
    """Return the explicitly supplied Gaussian-twin asset path.

    The asset contains a large third-party Gaussian capture and is therefore
    intentionally not part of the Newton source or ``newton-assets`` package.
    """
    asset_path = path or os.environ.get("NEWTON_GAUSSIAN_TWIN_ASSET")
    if not asset_path:
        raise ValueError(
            "This example requires a packaged Gaussian-twin USD asset. Pass "
            "--asset PATH or set NEWTON_GAUSSIAN_TWIN_ASSET."
        )
    return asset_path


@dataclass(frozen=True)
class SkinBinding:
    """Linear-blend skinning of a Gaussian field to a simulation mesh, baked into the asset.

    Attributes:
        influence_indices: Simulation mesh vertices driving each Gaussian, shape ``[num_gaussians, num_influences]``.
        influence_weights: Blend weight of each influence, shape ``[num_gaussians, num_influences]``.
        points: Rest positions of the simulation mesh [m], shape ``[num_vertices, 3]``.
        tet_indices: Vertices of each tetrahedron, shape ``[num_tets, 4]``.
        particle_radius: Contact radius authored with the mesh [m], or ``None`` when the
            asset does not carry one. VBD collides a soft body as one sphere per vertex, so
            this is what seals the boundary; a mesh whose vertex spacing is uniform can
            state a radius that a rule of thumb based on edge length would get wrong.
    """

    influence_indices: np.ndarray
    influence_weights: np.ndarray
    points: np.ndarray
    tet_indices: np.ndarray
    particle_radius: float | None = None


def _optional_float(prim, name: str) -> float | None:
    """Return an authored float attribute, or ``None`` when the asset omits it."""
    attr = prim.GetAttribute(name)
    return None if not attr or attr.Get() is None else float(attr.Get())


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
        particle_radius=_optional_float(tet_prim, "newton:simulationMesh:particleRadius"),
    )
    if len(binding.influence_indices) != count or len(binding.influence_weights) != count:
        raise ValueError(f"skinning binding does not match the authored pointCount of {count}")
    if binding.influence_indices.min() < 0 or binding.influence_indices.max() >= len(points):
        raise ValueError("skinning influences do not index the simulation mesh vertices")
    return binding


def tet_rest_bases(points: np.ndarray, tet_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the per-tetrahedron quantities the deformation gradient pass needs.

    Returns the inverse rest edge basis per tetrahedron [1/m], the rest volume
    each tetrahedron contributes to its vertices [m^3], and the reciprocal of the
    volume accumulated per vertex [1/m^3]. Degenerate tetrahedra (the packaged
    meshes contain a few slivers) get a zero basis and zero volume, so they drop
    out of the vertex average instead of polluting it.
    """
    corners = points[tet_indices]
    edges = np.stack([corners[:, k] - corners[:, 0] for k in (1, 2, 3)], axis=-1)
    determinant = np.linalg.det(edges)
    usable = np.abs(determinant) > 1.0e-16
    basis_inv = np.zeros_like(edges)
    basis_inv[usable] = np.linalg.inv(edges[usable])
    volume = np.where(usable, np.abs(determinant) / 6.0, 0.0)

    accumulated = np.zeros(len(points))
    np.add.at(accumulated, tet_indices.ravel(), np.repeat(volume, 4))
    inv_accumulated = np.where(accumulated > 0.0, 1.0 / np.maximum(accumulated, 1.0e-30), 0.0)
    return (
        basis_inv.astype(np.float32),
        volume.astype(np.float32),
        inv_accumulated.astype(np.float32),
    )


def fit_asset_to_simulation_transform(
    asset_points: np.ndarray, simulation_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the asset-to-simulation affine transform from matching TetMesh vertices.

    USD importers may apply the stage's axis and unit metadata before the
    caller's placement transform.  The baked Gaussian binding, on the other
    hand, is authored in the TetMesh's asset coordinates.  Deriving the
    transform from the imported TetMesh keeps both representations in the same
    space for any valid USD package instead of relying on a particular stage
    convention.

    Returns:
        The row-vector linear transform, translation, and its closest proper
        rotation (as a column-vector matrix).

    Raises:
        ValueError: If the importer changed vertex count, order, or geometry.
    """
    asset_points = np.asarray(asset_points, dtype=np.float64)
    simulation_points = np.asarray(simulation_points, dtype=np.float64)
    if asset_points.shape != simulation_points.shape or asset_points.ndim != 2 or asset_points.shape[1] != 3:
        raise ValueError(
            "The imported TetMesh does not have the same ordered vertices as the Gaussian skinning binding."
        )

    augmented = np.column_stack((asset_points, np.ones(len(asset_points))))
    affine, _, rank, _ = np.linalg.lstsq(augmented, simulation_points, rcond=None)
    if rank != 4:
        raise ValueError("The TetMesh vertices cannot determine an asset-to-simulation transform.")
    linear = affine[:3]
    translation = affine[3]
    reconstructed = asset_points @ linear + translation
    scale = max(float(np.ptp(simulation_points, axis=0).max()), 1.0e-6)
    max_error = float(np.max(np.abs(reconstructed - simulation_points)))
    if max_error > 1.0e-4 * scale:
        raise ValueError(
            "The USD importer changed TetMesh geometry rather than applying a single stage/placement transform; "
            "the authored Gaussian skinning binding cannot be used safely."
        )

    # ``linear`` maps row vectors.  Quaternion matrices map column vectors, so
    # transpose before extracting the nearest rotation and discard any uniform
    # stage scale from the orientation update.
    u, _, vh = np.linalg.svd(linear.T)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return linear.astype(np.float32), translation.astype(np.float32), rotation.astype(np.float32)


def measure_toy(points: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    """Return rig dimensions and the waist/ground reference from simulation-space points."""
    extent = np.ptp(points, axis=0)
    length, width, thickness = (float(value) for value in extent)
    middle_x = 0.5 * float(points[:, 0].min() + points[:, 0].max())
    waist = np.abs(points[:, 0] - middle_x) < 0.15 * length
    waist_y = points[waist, 1]
    waist_center_y = 0.5 * float(waist_y.min() + waist_y.max())
    waist_half_width = 0.5 * float(waist_y.max() - waist_y.min())
    ground_z = float(points[:, 2].min())
    return length, width, thickness, middle_x, waist_center_y, ground_z, waist_half_width


def mean_tet_edge_length(points: np.ndarray, tet_indices: np.ndarray) -> float:
    """Return the mean edge length of a tetrahedral mesh in its current units."""
    corners = points[tet_indices]
    return float(
        np.mean(
            [
                np.linalg.norm(corners[:, a] - corners[:, b], axis=1)
                for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ]
        )
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
    rest_scale: wp.array[wp.vec3],
    deform_scales: int,
    transforms_out: wp.array[wp.transform],
    scales_out: wp.array[wp.vec3],
):
    """Update Gaussian splat centers, rotations, and optional radii from VBD.

    Centers follow the weighted blend of the nodal displacements of the bound
    vertices, which reproduces the authored center exactly at rest even where
    the baked weights extrapolate outside the mesh. Orientations follow the
    rotation part of the deformation gradient blended with the same weights,
    which stays smooth across element boundaries. When requested, the three
    radii use the lengths of that gradient's material-axis images; Gaussian
    covariance cannot represent the remaining shear exactly.
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
    # orthonormalization of its columns.
    c0 = wp.vec3(gradient[0, 0], gradient[1, 0], gradient[2, 0])
    c1 = wp.vec3(gradient[0, 1], gradient[1, 1], gradient[2, 1])
    if wp.length(c0) < 1.0e-9 or wp.length(wp.cross(c0, c1)) < 1.0e-12:
        # Vertices without a usable gradient (no incident tetrahedron, or a
        # collapsed neighborhood) keep the bind-pose orientation.
        transforms_out[i] = wp.transform(center, rest_rotation[i])
        if deform_scales != 0:
            scales_out[i] = rest_scale[i]
        return

    c0 = wp.normalize(c0)
    c1 = wp.normalize(c1 - c0 * wp.dot(c0, c1))
    c2 = wp.cross(c0, c1)

    q_rot = wp.quat_from_matrix(wp.matrix_from_cols(c0, c1, c2))
    transforms_out[i] = wp.transform(center, wp.normalize(q_rot * rest_rotation[i]))
    if deform_scales != 0:
        # Gaussian covariance supports rotation and three principal radii, but
        # not shear.  The lengths of F's material-axis images retain its local
        # axial stretches while the orientation above carries its rotation.
        stretch = wp.vec3(
            wp.length(wp.vec3(gradient[0, 0], gradient[1, 0], gradient[2, 0])),
            wp.length(wp.vec3(gradient[0, 1], gradient[1, 1], gradient[2, 1])),
            wp.length(wp.vec3(gradient[0, 2], gradient[1, 2], gradient[2, 2])),
        )
        scale = rest_scale[i]
        scales_out[i] = wp.vec3(scale[0] * stretch[0], scale[1] * stretch[1], scale[2] * stretch[2])


@wp.kernel
def skin_gaussian_positions_to_mesh(
    particle_q: wp.array[wp.vec3],
    rest_particle_q: wp.array[wp.vec3],
    influence_indices: wp.array2d[wp.int32],
    influence_weights: wp.array2d[float],
    rest_position: wp.array[wp.vec3],
    rest_rotation: wp.array[wp.quat],
    transforms_out: wp.array[wp.transform],
):
    """Update Gaussian centers without evaluating deformation gradients."""
    i = wp.tid()
    center = rest_position[i]
    for k in range(influence_indices.shape[1]):
        v = influence_indices[i, k]
        center += (particle_q[v] - rest_particle_q[v]) * influence_weights[i, k]
    transforms_out[i] = wp.transform(center, rest_rotation[i])


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

    _MODE_ATTRIBUTES: ClassVar = {
        "position": ("positions",),
        "position-rotation": ("positions", "orientations"),
        "position-rotation-scale": ("positions", "orientations", "scales"),
    }

    def __init__(
        self,
        binding: SkinBinding,
        gaussian: newton.Gaussian,
        asset_linear: np.ndarray,
        asset_translation: np.ndarray,
        asset_rotation: np.ndarray,
        rest_particle_q: wp.array,
        deformation: str,
    ):
        device = rest_particle_q.device
        self.transforms = gaussian.warp_data.transforms
        self.scales = gaussian.warp_data.scales
        self.rest_particle_q = rest_particle_q
        self.deformation = deformation
        self.dynamic_attributes = self._MODE_ATTRIBUTES[deformation]
        # This is a private viewer hint: a general Gaussian asset does not know
        # which of its arrays an example changes.  It lets RTX stream exactly
        # the selected attributes instead of treating unchanged arrays as dirty.
        gaussian._newton_dynamic_attributes = self.dynamic_attributes

        # Fold the USD stage conversion and placement into the bind pose before
        # zeroing the shape transform.  This is inferred from the imported
        # TetMesh because USD axis/unit metadata is applied by the importer but
        # not to the custom skinning attributes.
        rest = self.transforms.numpy()
        rotation_quat = np.array(wp.quat_from_matrix(wp.mat33(*asset_rotation.reshape(-1).tolist())), dtype=np.float32)
        rest_position = rest[:, 0:3] @ asset_linear + asset_translation
        rest_rotation = quat_mul(rotation_quat, rest[:, 3:7])

        # The physics importer has already applied the asset's stage transform
        # and the example placement to ``rest_particle_q``. Measure the rest
        # bases in that same frame so the deformation gradient is identity at
        # bind pose; using the authored, untransformed basis would apply the
        # stage rotation a second time to every Gaussian orientation.
        basis_inv, tet_volume, vertex_inv_volume = tet_rest_bases(rest_particle_q.numpy(), binding.tet_indices)
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
            wp.clone(self.scales),
        ]

    def update(self, state: newton.State) -> None:
        """Skin the Gaussian field to the current particle positions."""
        device = self.transforms.device
        if self.deformation == "position":
            wp.launch(
                skin_gaussian_positions_to_mesh,
                dim=len(self.transforms),
                inputs=[state.particle_q, self.rest_particle_q, *self.skin_inputs[1:5]],
                outputs=[self.transforms],
                device=device,
            )
            return

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
            inputs=[
                state.particle_q,
                self.rest_particle_q,
                self.vertex_gradient,
                *self.skin_inputs,
                int(self.deformation == "position-rotation-scale"),
            ],
            outputs=[self.transforms, self.scales],
            device=device,
        )


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        if args.fast_simulation and args.balanced_simulation:
            raise ValueError("--fast-simulation and --balanced-simulation are mutually exclusive.")
        if args.fast_simulation:
            # Keep the material and contact gains intact.  Reducing them makes
            # the toy cheaper but also changes the demonstration into one
            # where the jaws cannot reliably retain it.  This preset only
            # reduces temporal resolution and nonlinear-solve convergence,
            # then lowers the drive bandwidth to match that resolution.
            args.substeps = FAST_SIMULATION_SUBSTEPS
            args.vbd_iterations = FAST_SIMULATION_VBD_ITERATIONS
            args.drive_frequency = FAST_SIMULATION_DRIVE_FREQUENCY
        elif args.balanced_simulation:
            # The midpoint preset retains twice the fast preset's temporal
            # resolution and 50% more nonlinear iterations. It is intended
            # for a visibly steadier grasp when the full conservative solve is
            # unnecessary, while its lower drive bandwidth remains resolvable.
            args.substeps = BALANCED_SIMULATION_SUBSTEPS
            args.vbd_iterations = BALANCED_SIMULATION_VBD_ITERATIONS
            args.drive_frequency = BALANCED_SIMULATION_DRIVE_FREQUENCY
        self.args = args
        self.viewer = viewer
        self.scene = args.scene
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.asset = resolve_asset(args.asset)
        binding = read_skin_binding(self.asset)
        # Turn the toy into the pose the jaws can actually pinch. A capture arrives in
        # whatever frame the scan had and a packaged toy is authored lying flat, while this
        # rig is built along fixed axes, so the pose is derived from the toy's own extents:
        # the longest onto +x, so the jaws close across the body rather than along it, the
        # *shortest* onto the closing direction +y, and what is left onto +z. The offset
        # then centers the toy over the origin.
        #
        # Pinching the shortest cross-section lets the jaws indent the toy
        # enough to retain it without requiring an excessive drive force.
        longest, middle, shortest = np.argsort(np.ptp(binding.points, axis=0))[::-1]
        basis = np.eye(3)[[longest, shortest, middle]]
        if np.linalg.det(basis) < 0.0:
            # An odd permutation mirrors the toy, so flip the closing axis to get a rotation
            # back. The jaws straddle the waist symmetrically and cannot tell the difference.
            basis[1] = -basis[1]
        lay_down = wp.quat_from_matrix(wp.mat33(*basis.flatten().tolist()))
        rest = binding.points @ basis.T.astype(np.float32)

        # Packaged toys differ in size and shape, so the rig and the motion schedule are sized
        # from the toy's own dimensions and from the waist, the cross-section the jaws close
        # on. The waist is what gets centered over the origin rather than the whole bounding
        # box: a captured toy is not symmetric, and centering the box leaves the waist off to
        # one side, where one jaw squeezes while the other closes on air.
        (
            self.toy_length,
            self.toy_width,
            self.toy_thickness,
            middle_x,
            waist_center_y,
            ground_z,
            self.waist_half_width,
        ) = measure_toy(rest)
        offset = np.array([-middle_x, -waist_center_y, -ground_z], dtype=np.float32)
        toy_xform = wp.transform(wp.vec3(*offset.tolist()), lay_down)
        edge_length = mean_tet_edge_length(binding.points, binding.tet_indices)
        # A radius authored with the mesh wins: the packaging step measures the boundary
        # vertex spacing and states a radius that makes neighbouring spheres overlap, which
        # a fraction of the mean edge length only approximates.
        if args.particle_radius > 0.0:
            self.particle_radius = args.particle_radius
        else:
            self.particle_radius = binding.particle_radius or 0.4 * edge_length

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder)

        # The asset authors no physics material, so the toy's material comes from the
        # builder defaults that the TetMesh import picks up.
        #
        # Express the default modulus as E/(rho*g*L), so differently sized
        # packaged toys sag by a comparable fraction of their length.
        self.youngs_modulus = args.youngs_modulus or (args.gravity_stiffness * args.density * 9.81 * self.toy_length)
        # A substep has to be short enough to resolve the fastest (longitudinal)
        # elastic wave crossing one tetrahedron, dt <= h / sqrt((lambda + 2*mu)/rho).
        # Derive the count from this transit time rather than assuming a
        # particular asset size or mesh resolution.
        poisson = args.poissons_ratio
        builder.default_tet_density = args.density
        builder.default_tet_k_mu = self.youngs_modulus / (2.0 * (1.0 + poisson))
        builder.default_tet_k_lambda = self.youngs_modulus * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        wave_speed = math.sqrt((builder.default_tet_k_lambda + 2.0 * builder.default_tet_k_mu) / args.density)
        self.sim_substeps = args.substeps or max(1, math.ceil(self.frame_dt * wave_speed / edge_length))
        self.sim_dt = self.frame_dt / self.sim_substeps
        # Viscous damping is a stress per unit strain rate, so the rate it imposes on an
        # element grows as the elements get smaller. Fixing the damping ratio against the
        # element's own elastic response instead keeps the toy equally damped at any mesh
        # resolution and toy size, where a fixed [Pa*s] value would diverge on fine meshes.
        builder.default_tet_k_damp = (
            2.0 * args.damping_ratio * edge_length * math.sqrt(self.youngs_modulus * args.density)
        )
        builder.default_particle_radius = self.particle_radius

        toy_particle_start = builder.particle_count
        toy_tet_start = len(builder.tet_materials)
        results = builder.add_usd(self.asset, xform=toy_xform, ignore_paths=IGNORE_PATHS)
        self.toy_particles = list(range(toy_particle_start, builder.particle_count))
        # The baked binding indexes the authored mesh.  USD import applies stage
        # axis/unit metadata and the requested placement to the physics mesh,
        # so infer that common transform from the ordered TetMesh vertices and
        # use it for the Gaussian bind pose below.
        imported = np.asarray(builder.particle_q[toy_particle_start:], dtype=np.float32)
        asset_linear, asset_translation, asset_rotation = fit_asset_to_simulation_transform(binding.points, imported)
        # All dimensions used by the rig and the material are physical, so
        # recompute them after the USD importer has applied stage units/axes
        # and the requested placement. The initial asset-space estimate above
        # exists only to construct that placement transform.
        (
            self.toy_length,
            self.toy_width,
            self.toy_thickness,
            self.toy_center_x,
            self.toy_center_y,
            self.toy_ground_z,
            self.waist_half_width,
        ) = measure_toy(imported)
        edge_length = mean_tet_edge_length(imported, binding.tet_indices)
        if args.particle_radius <= 0.0:
            if binding.particle_radius is not None:
                self.particle_radius = binding.particle_radius * float(np.cbrt(abs(np.linalg.det(asset_linear))))
            else:
                self.particle_radius = 0.4 * edge_length
            for particle in self.toy_particles:
                builder.particle_radius[particle] = self.particle_radius

        self.youngs_modulus = args.youngs_modulus or (args.gravity_stiffness * args.density * 9.81 * self.toy_length)
        builder.default_tet_k_mu = self.youngs_modulus / (2.0 * (1.0 + poisson))
        builder.default_tet_k_lambda = self.youngs_modulus * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        builder.default_tet_k_damp = (
            2.0 * args.damping_ratio * edge_length * math.sqrt(self.youngs_modulus * args.density)
        )
        for tet in range(toy_tet_start, len(builder.tet_materials)):
            builder.tet_materials[tet] = (
                builder.default_tet_k_mu,
                builder.default_tet_k_lambda,
                builder.default_tet_k_damp,
            )
        wave_speed = math.sqrt((builder.default_tet_k_lambda + 2.0 * builder.default_tet_k_mu) / args.density)
        self.sim_substeps = args.substeps or max(1, math.ceil(self.frame_dt * wave_speed / edge_length))
        self.sim_dt = self.frame_dt / self.sim_substeps
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
        particle_mass = float(np.mean(self.model.particle_mass.numpy()[self.toy_particles]))
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
            # A jaw closing on the toy can touch every one of its particles at once, so the
            # per-body soft-contact list has to be as long as the toy, not a fixed guess
            # that a finer simulation mesh silently overflows.
            "rigid_body_particle_contact_buffer_size": max(256, len(self.toy_particles)),
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

        self.gaussian_updates_enabled = not args.no_gaussian_update
        self.twin = None
        if self.gaussian_updates_enabled:
            self.twin = GaussianTwin(
                binding,
                self.model.shape_source[gaussian_shape],
                asset_linear,
                asset_translation,
                asset_rotation,
                wp.clone(self.state_0.particle_q),
                args.splat_deformation,
            )
            self.twin.update(self.state_0)
        self.initial_tet_center = np.mean(self.state_0.particle_q.numpy(), axis=0)
        self.initial_gaussian_center = (
            np.mean(self.twin.transforms.numpy()[:, 0:3], axis=0) if self.twin is not None else None
        )

        target_q_start = self.model.joint_target_q_start.numpy()
        self.lift_target = int(target_q_start[self.lift_joint])
        self.sway_target = int(target_q_start[self.sway_joint])
        self.finger_targets = (int(target_q_start[self.left_joint]), int(target_q_start[self.right_joint]))
        self.max_toy_swing = 0.0

        # Set this before set_model() enters the RTX build phase. The viewer
        # still receives the Gaussian shape as part of the model, but leaves
        # it hidden and never schedules per-frame Fabric writes.
        if not self.gaussian_updates_enabled and hasattr(self.viewer, "show_gaussians"):
            self.viewer.show_gaussians = False
        newton.examples.configure_coupled_view(self, args)
        # Only some backends draw Gaussian assets; ``log_gaussian`` is a no-op in the
        # base viewer. Without it the toy would be invisible, so fall back to the
        # simulation mesh surface and say why.
        # Viser's Gaussian API currently caches its immutable asset positions,
        # so it cannot visualize this field's per-frame skinning correctly.
        # Use the live tetmesh fallback there rather than showing a static twin.
        splats_supported = (
            self.gaussian_updates_enabled
            and type(self.viewer).log_gaussian is not ViewerBase.log_gaussian
            and not isinstance(self.viewer, ViewerViser)
        )
        if not splats_supported and not args.quiet:
            detail = (
                "Gaussian updates are disabled"
                if not self.gaussian_updates_enabled
                else f"{type(self.viewer).__name__} cannot render Gaussian splats"
            )
            print(
                f"{detail}; showing the simulation mesh instead. "
                "Run with '--viewer gl' or '--viewer rtx' without '--no-gaussian-update' to see the Gaussian twin."
            )
        if hasattr(self.viewer, "show_gaussians"):
            # The visible appearance is meant to come from the splats alone, so the tet
            # mesh surface stays hidden unless it is asked for or nothing else would show.
            self.viewer.show_gaussians = splats_supported
            self.viewer.show_triangles = args.show_tetmesh or not splats_supported
            if self.twin is not None:
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
        # closed, so both gaps come from the waist's own half width. The
        # closing target is a real gap, not a travel distance from the open gap.
        grip_z = self.toy_ground_z + self.args.grip_height * self.toy_thickness
        open_y = self.args.finger_open * self.waist_half_width
        self.finger_close = self.args.pinch_depth * self.waist_half_width
        back_x = self.toy_center_x - 0.7 * length
        hub = 0.06 * length
        finger_color = wp.vec3(0.88, 0.48, 0.22)

        carriage_xform = wp.transform(wp.vec3(back_x - 1.5 * hub, self.toy_center_y, grip_z), wp.quat_identity())
        carriage = builder.add_link(xform=carriage_xform, label="carriage")
        builder.add_shape_box(carriage, hx=hub, hy=hub, hz=hub, cfg=cfg, color=wp.vec3(0.3, 0.32, 0.36))
        palm = builder.add_link(
            xform=wp.transform(wp.vec3(back_x, self.toy_center_y, grip_z), wp.quat_identity()), label="palm"
        )
        builder.add_shape_box(palm, hx=hub, hy=open_y, hz=hub, cfg=cfg, color=wp.vec3(0.22, 0.28, 0.34))
        finger_hx = 0.18 * length
        finger_hy = 0.02 * length
        finger_hz = 0.5 * self.toy_thickness
        left = builder.add_link(
            xform=wp.transform(wp.vec3(self.toy_center_x, self.toy_center_y + open_y, grip_z), wp.quat_identity()),
            label="left",
        )
        builder.add_shape_box(left, hx=finger_hx, hy=finger_hy, hz=finger_hz, cfg=cfg, color=finger_color)
        right = builder.add_link(
            xform=wp.transform(wp.vec3(self.toy_center_x, self.toy_center_y - open_y, grip_z), wp.quat_identity()),
            label="right",
        )
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
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._update_targets()
        if self.graph is not None:
            with wp.ScopedDevice(self.model.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        if self.twin is not None:
            self.twin.update(self.state_0)
        # This validation is only needed by the test harness.  ``numpy()`` is a
        # device-wide synchronization and copying all particles every interactive
        # frame prevents simulation and rendering from overlapping.
        if self.args.test:
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
        # A toy the jaws are holding stays with them, so the travel is bounded on both
        # sides. Only asking for a lower bound let a run through in which the toy was
        # flung 6 metres, a swing of 175 times the amplitude the carriage was driven at.
        reach = (self.args.sway_distance if self.scene == "sway" else 0.0) + 1.0
        assert self.max_toy_swing < reach * length, (
            f"Toy was thrown rather than carried: max dx={self.max_toy_swing:.4f}"
        )
        # Short runs (the CI smoke test uses two frames) stop before the gripper has
        # moved, so only check the outcome once the motion schedule has played out.
        lift_done = self.args.close_time + self.args.lift_time
        if self.sim_time > lift_done:
            assert tet_shift[2] > 0.15 * length, f"Gripper did not lift the toy: dz={tet_shift[2]:.4f}"
        if self.scene == "sway" and self.sim_time > lift_done + 0.5 * self.args.sway_period:
            assert self.max_toy_swing > 0.06 * length, (
                f"Carriage did not swing the toy: max dx={self.max_toy_swing:.4f}"
            )

        if self.twin is not None:
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
            "Required unless NEWTON_GAUSSIAN_TWIN_ASSET is set.",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=0,
            help="Simulation substeps per rendered frame. Defaults to the count that keeps a substep "
            "inside the time an elastic wave takes to cross one tetrahedron, which is what makes a "
            "small toy behave like a large one.",
        )
        parser.add_argument("--vbd-iterations", type=int, default=60, help="VBD iterations per substep.")
        parser.add_argument(
            "--fast-simulation",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use the interactive Gaussian-twin preset (4 substeps x 30 VBD iterations, and a 7.5 Hz "
            "gripper drive, per rendered frame). "
            "It reduces the default solver work by roughly 10x and preserves the packaged toy's grasp/lift "
            "demonstration, but is less accurate for stiff material waves and contact forces. This preset "
            "intentionally overrides --substeps, --vbd-iterations, and --drive-frequency.",
        )
        parser.add_argument(
            "--balanced-simulation",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use the balanced Gaussian-twin preset (8 substeps x 45 VBD iterations and a 12.5 Hz "
            "gripper drive per rendered frame). It sits between the conservative default and "
            "--fast-simulation: steadier grasp/contact behavior than the fast preset at about three times its "
            "solver work. This preset intentionally overrides --substeps, --vbd-iterations, and "
            "--drive-frequency; it cannot be combined with --fast-simulation.",
        )
        parser.add_argument("--density", type=float, default=300.0, help="Toy density [kg/m^3].")
        parser.add_argument(
            "--youngs-modulus",
            type=float,
            default=0.0,
            help="Toy Young's modulus [Pa]. Defaults to whatever --gravity-stiffness implies for the toy's own size.",
        )
        parser.add_argument(
            "--gravity-stiffness",
            type=float,
            default=125.0,
            help="Toy stiffness as a multiple of its own weight per unit area, E/(rho*g*L). Dimensionless, "
            "so the toy sags the same fraction of its length at any authored size.",
        )
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
            default=0.9,
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
        parser.add_argument(
            "--splat-deformation",
            choices=GaussianTwin._MODE_ATTRIBUTES,
            default="position-rotation-scale",
            help="Gaussian attributes deformed from the tet mesh: centers only, centers plus orientation, or "
            "centers plus orientation and per-axis scale. Scale deformation approximates stretch but cannot "
            "represent shear.",
        )
        parser.add_argument(
            "--no-gaussian-update",
            action="store_true",
            help="Disable Gaussian skinning and RTX Gaussian streaming, and show the simulation tetmesh instead. "
            "Use this to measure physics without per-frame Gaussian update or rendering cost.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
