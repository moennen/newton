# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM Pinch and Carry
#
# Picks up a soft fruit the way a parallel-jaw gripper does, to show a
# limitation of SolverImplicitMPM for robotic picking. A raspberry scanned
# as a 3D Gaussian splat carries its own volumetric tissue: about 3,000
# particles at 1 mm spacing, simulated as an elastic MPM solid. Two
# kinematic box pads close on it to 70% of its width, lift it 50 mm, hold
# it, and carry it 100 mm sideways.
#
# The lift succeeds; the carry does not. While the pads hold their gap,
# the stress with which the tissue pushes back on them drains away, even
# though its stored elastic deformation stays compressed and even grows.
# Once the normal stress is gone, collider friction has nothing to act on
# and the berry drops. With the defaults, the squeeze force falls from about
# 0.07 N at the grasp to under 0.02 N by the end of the lift, and the berry
# falls within 0.3 s of the carry starting. Pad friction 20 only delays the
# drop to mid-carry; yield limits of 100 times Young's modulus change
# nothing; 250 iterations start with a firmer grip that drains the same way.
# At 240 Hz (--substeps 4) the berry is not even lifted.
#
# Every tenth of a second the example prints the berry height relative to
# the pads, the squeeze force on the pads (from the collider impulses), and,
# for the particles next to the pads, the mean solver stress sigma_yy and the
# stored elastic compression Fe_yy - 1 along the squeeze axis. A summary per
# phase follows at the end.
#
# The asset is not distributed with Newton. Pass --asset or set
# NEWTON_BERRY_ASSET to a berry USD package with a /Berry/Tissue points prim
# (tissue positions) and, for --show-gaussians, a /Berry/Gaussians splat.
#
# Command: python -m newton.examples mpm_pinch_carry --asset path/to/raspberry_v2.usdz
#          python -m newton.examples mpm_pinch_carry --asset path/to/raspberry_v2.usdz --pad-friction 20
#          python -m newton.examples mpm_pinch_carry --asset path/to/raspberry_v2.usdz --viewer null --num-frames 420
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

# Pad half extents [m]: 18 mm wide (x), 10 mm thick (y, the squeeze axis), 18 mm tall (z).
PAD_HALF_EXTENTS = (0.009, 0.005, 0.009)

# Scripted gripper motion: (phase name, duration [s], gap [fraction of the berry width], lift [m], carry [m]).
# Each phase moves linearly from the previous phase's end values to its own.
PHASES = (
    ("settle", 0.5, 1.4, 0.0, 0.0),
    ("close", 1.0, 0.7, 0.0, 0.0),
    ("hold", 0.5, 0.7, 0.0, 0.0),
    ("lift", 1.0, 0.7, 0.05, 0.0),
    ("hold high", 0.5, 0.7, 0.05, 0.0),
    ("carry", 2.0, 0.7, 0.05, 0.1),
    ("hold carried", 1.0, 0.7, 0.05, 0.1),
)


@wp.kernel
def advance_time(time: wp.array[float], dt: float):
    time[0] = time[0] + dt


@wp.kernel
def move_pads(
    time: wp.array[float],
    phase_end: wp.array[float],
    gap: wp.array[float],
    lift: wp.array[float],
    carry: wp.array[float],
    center: wp.vec3,
    carry_axis: wp.vec3,
    half_thickness: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Pose both pads on the piecewise-linear script; pad 0 sits at -y, pad 1 at +y."""
    pad = wp.tid()
    t = time[0]
    k = int(0)
    while k < phase_end.shape[0] - 1 and t > phase_end[k]:
        k += 1
    start = float(0.0)
    if k > 0:
        start = phase_end[k - 1]
    duration = phase_end[k] - start
    u = wp.clamp((t - start) / duration, 0.0, 1.0)
    gap_k = wp.lerp(gap[k], gap[k + 1], u)
    lift_k = wp.lerp(lift[k], lift[k + 1], u)
    carry_k = wp.lerp(carry[k], carry[k + 1], u)
    rate = wp.where(t < phase_end[phase_end.shape[0] - 1], 1.0 / duration, 0.0)
    side = wp.where(pad == 0, -1.0, 1.0)
    offset = wp.vec3(0.0, side * (0.5 * gap_k + half_thickness), lift_k)
    body_q[pad] = wp.transform(center + offset + carry_k * carry_axis, wp.quat_identity())
    velocity = wp.vec3(0.0, side * 0.5 * (gap[k + 1] - gap[k]), lift[k + 1] - lift[k])
    velocity = (velocity + (carry[k + 1] - carry[k]) * carry_axis) * rate
    body_qd[pad] = wp.spatial_vector(velocity, wp.vec3(0.0))


@wp.kernel
def follow_nearest(
    rest: wp.array[wp.vec3],
    nearest: wp.array[int],
    particle_rest: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    p = nearest[i]
    out[i] = rest[i] + particle_q[p] - particle_rest[p]


class Example:
    """A raspberry pinched by two kinematic pads, lifted and carried with SolverImplicitMPM."""

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.viewer = viewer

        asset = args.asset or os.environ.get("NEWTON_BERRY_ASSET")
        if not asset:
            raise ValueError("Pass --asset or set NEWTON_BERRY_ASSET to a berry USD package")
        tissue, spacing, gaussians = Example.load_berry(asset, args.show_gaussians)
        volume = spacing**3
        lower, upper = tissue.min(0), tissue.max(0)
        self.width = float(upper[1] - lower[1])
        center = 0.5 * (lower + upper)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_ground_plane()
        radius = 0.5 * spacing
        for p in tissue:
            builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=args.density * volume, radius=radius)
        pad_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.pad_friction)
        self.pads = []
        for side in ("left", "right"):
            body = builder.add_body(is_kinematic=True, label=f"pad_{side}")
            builder.add_shape_box(
                body, hx=PAD_HALF_EXTENTS[0], hy=PAD_HALF_EXTENTS[1], hz=PAD_HALF_EXTENTS[2], cfg=pad_cfg
            )
            self.pads.append(body)
        self.model = builder.finalize()

        mpm = self.model.mpm
        mpm.young_modulus.fill_(args.young_modulus)
        mpm.poisson_ratio.fill_(args.poisson_ratio)
        mpm.friction.fill_(args.tissue_friction)
        mpm.yield_stress.fill_(args.yield_ratio * args.young_modulus)
        mpm.yield_pressure.fill_(args.yield_pressure_ratio * args.young_modulus)
        mpm.tensile_yield_ratio.fill_(1.0)

        config = SolverImplicitMPM.Config()
        config.voxel_size = args.voxel_size if args.voxel_size > 0.0 else 2.0 * spacing
        config.grid_type = "sparse"
        config.grid_padding = 0
        config.max_active_cell_count = args.max_active_cells
        config.strain_basis = args.strain_basis
        config.transfer_scheme = "apic"
        config.max_iterations = args.max_iterations
        config.tolerance = args.tolerance
        config.velocity_basis = "Q1"
        config.collider_basis = args.collider_basis
        config.collider_velocity_mode = "forward"
        config.solver = args.solver
        self.solver = SolverImplicitMPM(self.model, config)

        # The gripper closes on the berry's middle, across its y width, and carries it along --carry-axis.
        axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0)}[args.carry_axis]
        self.pad_center = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
        self.carry_axis = wp.vec3(*axis)
        ends = np.cumsum([duration for _, duration, *_ in PHASES])
        self.phase_names = [name for name, *_ in PHASES]
        self.phase_end = wp.array(ends, dtype=float)
        self.gap = wp.array([PHASES[0][2] * self.width] + [g * self.width for *_, g, _, _ in PHASES], dtype=float)
        self.lift = wp.array([0.0] + [lift for *_, lift, _ in PHASES], dtype=float)
        self.carry = wp.array([0.0] + [carry for *_, carry in PHASES], dtype=float)
        self.time = wp.zeros(1, dtype=float)
        self.duration = float(ends[-1])

        self.state = self.model.state()
        self.solver.reset(self.state)
        self.particle_rest = wp.clone(self.state.particle_q)
        self.move_pads(self.state)
        self.records: list[dict] = []

        self.gaussians = None
        if gaussians is not None:
            self.gaussians = Example.bind_gaussians(gaussians, tissue, self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(0.05, -0.25, 0.12), pitch=-20.0, yaw=90.0)

        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def move_pads(self, state):
        wp.launch(
            move_pads,
            dim=2,
            inputs=[
                self.time,
                self.phase_end,
                self.gap,
                self.lift,
                self.carry,
                self.pad_center,
                self.carry_axis,
                PAD_HALF_EXTENTS[1],
            ],
            outputs=[state.body_q, state.body_qd],
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.move_pads(self.state)
            self.solver.step(self.state, self.state, None, None, self.sim_dt)
            wp.launch(advance_time, dim=1, inputs=[self.time, self.sim_dt])

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.solver.check_sparse_grid_rebuild_status()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self.frame % max(1, round(0.1 * self.fps)) == 0:
            self.record()

    def phase(self, t: float) -> str:
        ends = self.phase_end.numpy()
        return self.phase_names[min(int(np.searchsorted(ends, t)), len(ends) - 1)]

    def record(self):
        """Measure the grip and print one line of the log."""
        q = self.state.particle_q.numpy()
        pads = self.state.body_q.numpy()[self.pads, :3]
        impulses, _, colliders = self.solver.collect_collider_impulses(self.state)
        body = self.solver.collider_body_index.numpy()[np.maximum(colliders.numpy(), 0)]
        on_pads = (colliders.numpy() >= 0) & np.isin(body, self.pads)
        squeeze = 0.5 * np.abs(impulses.numpy()[on_pads, 1]).sum() / self.sim_dt

        # Particles within 3 mm of either pad's inner face.
        inner = (pads[0, 1] + PAD_HALF_EXTENTS[1], pads[1, 1] - PAD_HALF_EXTENTS[1])
        near = (np.abs(q[:, 1] - inner[0]) < 0.003) | (np.abs(q[:, 1] - inner[1]) < 0.003)
        stress = self.state.mpm.particle_stress.numpy()[near, 1, 1]
        elastic = self.state.mpm.particle_elastic_strain.numpy()[near, 1, 1] - 1.0
        row = {
            "time": self.sim_time,
            "phase": self.phase(self.sim_time),
            "berry_height": float(q[:, 2].mean()),
            "relative_height": float(q[:, 2].mean() - pads[:, 2].mean()),
            "squeeze": float(squeeze),
            "stress": float(stress.mean()) if len(stress) else 0.0,
            "compression": float(elastic.mean()) if len(elastic) else 0.0,
        }
        self.records.append(row)
        print(
            f"t={row['time']:5.2f} s  {row['phase']:<12s}  berry z={1000 * row['berry_height']:6.1f} mm  "
            f"vs pads {1000 * row['relative_height']:+6.1f} mm  squeeze {row['squeeze']:7.3f} N  "
            f"sigma_yy {row['stress']:+8.1f} Pa  Fe_yy-1 {row['compression']:+.3f}",
            flush=True,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.gaussians is not None:
            g = self.gaussians
            wp.launch(
                follow_nearest,
                dim=len(g["rest"]),
                inputs=[g["rest"], g["nearest"], self.particle_rest, self.state.particle_q],
                outputs=[g["positions"]],
            )
            self.viewer.log_points("gaussians", g["positions"], g["radii"], g["colors"])
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_particle_state(
            self.state, "all particles have finite positions", lambda q, qd: wp.length(q) < 1.0e3
        )

    def summarize(self):
        """Print the grip at the end of each phase and whether the berry was carried."""
        if not self.records:
            return
        print("\nphase          end [s]  berry vs pads [mm]  squeeze [N]  sigma_yy [Pa]  Fe_yy-1")
        for name in self.phase_names:
            rows = [r for r in self.records if r["phase"] == name]
            if rows:
                r = rows[-1]
                print(
                    f"{name:<12s}  {r['time']:7.2f}  {1000 * r['relative_height']:+18.1f}  {r['squeeze']:11.3f}  "
                    f"{r['stress']:+13.1f}  {r['compression']:+.3f}"
                )
        lifted = [r for r in self.records if r["phase"] in ("hold high", "carry", "hold carried")]
        dropped = next((r for r in lifted if r["relative_height"] < -0.01), None)
        if dropped is None:
            print("\nResult: the berry stayed between the pads.")
        else:
            print(f"\nResult: the berry dropped out of the grip at t={dropped['time']:.2f} s ({dropped['phase']}).")

    @staticmethod
    def load_berry(path: str, with_gaussians: bool):
        from pxr import Usd  # noqa: PLC0415

        stage = Usd.Stage.Open(path)
        tissue = stage.GetPrimAtPath("/Berry/Tissue")
        if not tissue:
            raise ValueError(f"{path} has no /Berry/Tissue points prim")
        points = np.asarray(tissue.GetAttribute("points").Get(), dtype=np.float32)
        spacing_attr = tissue.GetAttribute("berry:spacing")
        spacing = float(spacing_attr.Get()) if spacing_attr else 0.001
        gaussians = None
        if with_gaussians:
            prim = stage.GetPrimAtPath("/Berry/Gaussians")
            gaussians = newton.Gaussian.create_from_usd(prim) if prim else None
        return points, spacing, gaussians

    @staticmethod
    def bind_gaussians(gaussians: newton.Gaussian, tissue: np.ndarray, device) -> dict:
        """Attach each Gaussian center to its nearest tissue particle, for display only."""
        centers = np.asarray(gaussians.positions, dtype=np.float32)
        nearest = np.empty(len(centers), dtype=np.int32)
        for start in range(0, len(centers), 4096):
            chunk = centers[start : start + 4096]
            nearest[start : start + 4096] = np.argmin(((chunk[:, None, :] - tissue[None, :, :]) ** 2).sum(-1), axis=1)
        sh = gaussians.sh_coeffs
        dc = np.asarray(sh, dtype=np.float32).reshape(len(centers), -1, 3)[:, 0] if sh is not None else None
        colors = np.clip(0.5 + 0.28209479 * dc, 0.0, 1.0) if dc is not None else np.full((len(centers), 3), 0.8)
        return {
            "rest": wp.array(centers, dtype=wp.vec3, device=device),
            "nearest": wp.array(nearest, dtype=int, device=device),
            "positions": wp.empty(len(centers), dtype=wp.vec3, device=device),
            "radii": wp.full(len(centers), 0.0003, dtype=float, device=device),
            "colors": wp.array(colors.astype(np.float32), dtype=wp.vec3, device=device),
        }

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--asset", type=str, default=None, help="Berry USD package (or NEWTON_BERRY_ASSET).")
        parser.add_argument("--show-gaussians", action="store_true", help="Draw the splat centers on the tissue.")
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=8, help="Solver steps per frame (480 Hz by default).")
        parser.add_argument("--carry-axis", choices=["x", "y"], default="y", help="y is the squeeze axis.")
        parser.add_argument("--pad-friction", type=float, default=1.0)
        parser.add_argument("--tissue-friction", type=float, default=0.35)
        parser.add_argument("--density", type=float, default=900.0)
        parser.add_argument("--young-modulus", "-ym", type=float, default=12000.0)
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.4)
        parser.add_argument("--yield-ratio", type=float, default=0.5, help="Shear yield stress over Young's modulus.")
        parser.add_argument("--yield-pressure-ratio", type=float, default=5.0)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.0, help="0 picks twice the tissue spacing.")
        parser.add_argument("--max-active-cells", type=int, default=4096)
        parser.add_argument("--strain-basis", "-sb", type=str, default="P1d")
        parser.add_argument("--collider-basis", type=str, default="pic27")
        parser.add_argument("--max-iterations", "-it", type=int, default=100)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
        parser.add_argument("--solver", "-s", type=str, default="jacobi")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.summarize()
