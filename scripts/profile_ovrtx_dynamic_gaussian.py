#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture a steady-state dynamic-Gaussian OVRTX workload.

This intentionally small runner is an NVBug repro harness, not an application
benchmark. It warms up the simulation, OVRTX stage, and GPU bindings before it
marks a finite sequence of frames with NVTX. By default it synchronizes after
the simulation step so OVRTX update time cannot include unfinished Newton
simulation work from the same frame.

Example:
    NEWTON_PROFILE=0 nsys profile --trace=cuda,vulkan,nvtx,osrt \\
        --vulkan-gpu-workload=individual --force-overwrite=true \\
        --output=/tmp/ovrtx-gaussian \\
        uv run python scripts/profile_ovrtx_dynamic_gaussian.py \\
        --asset /path/to/package.usda
"""

from __future__ import annotations

import argparse
import ctypes
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import warp as wp

from newton.examples.multiphysics.example_mujoco_vbd_gaussian_twin import Example
from newton.viewer import ViewerRTX


class Nvtx:
    """Minimal NVTX binding that does not add a Python package dependency."""

    def __init__(self) -> None:
        self._library = ctypes.CDLL("libnvToolsExt.so.1")
        self._library.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self._library.nvtxRangePushA.restype = ctypes.c_int
        self._library.nvtxRangePop.argtypes = []
        self._library.nvtxRangePop.restype = ctypes.c_int

    @contextmanager
    def range(self, label: str):
        """Emit an NVTX range with balanced cleanup on an exception."""
        self._library.nvtxRangePushA(label.encode())
        try:
            yield
        finally:
            self._library.nvtxRangePop()


def _parse_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Fully rendered frames before the marked capture interval.",
    )
    profile_parser.add_argument(
        "--capture-frames",
        type=int,
        default=120,
        help="Marked dynamic-Gaussian frames to execute after warm-up.",
    )
    profile_parser.add_argument(
        "--wait-for-file",
        type=Path,
        default=None,
        help="After warm-up, wait until this file exists before starting the marked interval.",
    )
    profile_parser.add_argument(
        "--no-step-sync",
        dest="synchronize_step",
        action="store_false",
        help="Allow OVRTX's Fabric dependency to consume the simulation-to-renderer wait.",
    )
    profile_parser.set_defaults(synchronize_step=True)
    profile_args, example_args = profile_parser.parse_known_args()
    if profile_args.warmup_frames < 0 or profile_args.capture_frames < 1:
        profile_parser.error("--warmup-frames must be non-negative and --capture-frames must be positive")

    parser = Example.create_parser()
    parser.set_defaults(viewer="rtx")
    args = parser.parse_args(example_args)
    if not args.asset:
        parser.error("--asset is required for this profiling harness")
    if args.viewer != "rtx":
        parser.error("this profiling harness always uses ViewerRTX; do not override --viewer rtx")
    args.headless = True
    args.num_frames = profile_args.warmup_frames + profile_args.capture_frames + 1
    return profile_args, args


def _instrument_viewer(viewer: ViewerRTX, nvtx: Nvtx) -> None:
    """Wrap the public renderer stages without changing the workload."""

    def wrap(method: Callable[..., Any], label: str) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with nvtx.range(label):
                return method(*args, **kwargs)

        return wrapped

    for method_name in (
        "_update_ovrtx_transforms",
        "_update_ovrtx_instance_visibility",
        "_update_ovrtx_gaussians",
        "_render_and_display",
    ):
        method = getattr(viewer, method_name)
        setattr(viewer, method_name, wrap(method, f"ViewerRTX::{method_name}"))


def _wait_for_capture_start(path: Path) -> None:
    print(f"warm state reached; waiting for {path}", flush=True)
    while not path.exists():
        time.sleep(0.1)
    print("starting marked dynamic-Gaussian interval", flush=True)


def main() -> None:
    profile_args, args = _parse_args()
    nvtx = Nvtx()
    viewer = ViewerRTX(headless=True, async_rendering=False, num_frames=args.num_frames)
    example = Example(viewer, args)
    _instrument_viewer(viewer, nvtx)

    try:
        for _ in range(profile_args.warmup_frames):
            example.step()
            example.render()

        if profile_args.wait_for_file is not None:
            _wait_for_capture_start(profile_args.wait_for_file)
        else:
            print("warm state reached; starting marked dynamic-Gaussian interval", flush=True)

        with nvtx.range("OVRTX Dynamic Gaussian Rebuild"):
            for frame in range(profile_args.capture_frames):
                with nvtx.range(f"frame {frame:03d}"):
                    with nvtx.range("Newton::step_submit"):
                        example.step()
                    if profile_args.synchronize_step:
                        with nvtx.range("Newton::step_gpu_sync"):
                            wp.synchronize()
                    with nvtx.range("Newton::render"):
                        example.render()
    finally:
        viewer.close()

    print(f"captured {profile_args.capture_frames} dynamic-Gaussian frames", flush=True)


if __name__ == "__main__":
    main()
