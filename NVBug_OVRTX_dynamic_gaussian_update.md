# OVRTX 0.4.1: dynamic Gaussian position updates stall `update()`

## Summary

With OVRTX 0.4.1, updating only `points` on an existing
`ParticleField3DGaussianSplat` every frame can make
`HydraEngineWrapper::update()` take roughly 100 ms on the CPU. The measured
RTX tile GPU work is about 15 ms, so the stall is not explained by ray tracing.

The supplied Newton harness uses the released OVRTX 0.4.1 wheel and a Gaussian
USD package with a tetmesh skinning binding. It simulates the tetmesh on CUDA,
skins the Gaussian centers on CUDA, then submits a GPU `points` array update to
the existing ParticleField. Orientation, scale, opacity, SH degree, and SH
coefficients remain unchanged.

This is a steady-state issue: capture starts after OVRTX has opened the stage,
the Gaussian binding has been established, and 30 fully rendered frames have
completed. It is therefore distinct from shader compilation and initial asset
loading.

## Environment

| Component | Version |
| --- | --- |
| OVRTX | 0.4.1 (released wheel, no local OVRTX binary changes) |
| Newton | branch `nicolasm/ovrtx-gaussian-profiling-repro` |
| Warp | 1.17.0 |
| CUDA toolkit / driver | 12.9 / 13.2 |
| GPU | NVIDIA RTX 6000 Ada Generation, 47 GiB |
| Nsight Systems | 2026.4.1 |

## Observed behavior

A prior Tracy capture of the same workload, during active simulation and after
scene initialization, measured:

| Scope | Mean CPU time per update | Notes |
| --- | ---: | --- |
| `HydraEngineWrapper::update` | 103.2 ms | 158 updates; 1.7--147.4 ms range |
| Fabric `ParticleField` primvar refresh | 9.5 ms | Re-reads static primvars after a `points` update |
| `ParticleField` SH-coefficient read | 9.0 ms | Part of the preceding refresh |
| OVRTX `writeAttributeToStageImpl_array_gpu` | 11.0 ms | CPU elapsed time in the Fabric GPU-array write path |
| GPU `RTX Render Tile` | 15.5 ms | 153 tiles; 8.8--23.2 ms range |

The remaining main-thread interval is an unscoped wait inside the Hydra update
path. A Carbonite tasking worker is active at the same time:

* tasking worker: `Running fiber` / `Executing task`, about 48 ms;
* nested `carb.ujitsoagent::storeExternalData`, about 22.6 ms;
* overlapping datastore-worker `LocalDataStore::Write back`, about 26.2 ms.

`BasePreprocessedGaussians::updateGaussianResources` is called on nearly every
dynamic position-only update. That correlation suggests that resource
preprocessing/cache work is re-entered for a deformation that should require
only changed Gaussian positions and its dynamic acceleration structure update.
It is an inference from the trace, not yet a proven call graph.

## Expected behavior

For a position-only update of a pre-existing Gaussian field, static Gaussian
primvars and preprocessed resources should remain reusable. CPU update time
should not be dominated by background UJITSO/datastore work, and it should be
closer to the GPU submission/render work plus the required dynamic Gaussian
acceleration-structure maintenance.

## Reproduction steps

1. Check out this branch and install the normal Newton example dependencies.

   ```bash
   cd /mnt/dev/isaaclab-galois/newton
   uv sync --extra examples
   uv pip install --reinstall 'ovrtx==0.4.1'
   ```

2. Confirm that no locally rebuilt OVRTX runtime is being used.

   ```bash
   uv run python -c 'import ovrtx; print(ovrtx.__version__)'
   ```

3. Run a bounded Nsight Systems capture. The explicit synchronization separates
   simulation completion from OVRTX update, so render/update time cannot be
   charged to unfinished Newton simulation work.

   ```bash
   NEWTON_PROFILE=0 nsys profile \
       --trace=cuda,vulkan,nvtx,osrt \
       --sample=process-tree \
       --cpuctxsw=process-tree \
       --cuda-event-trace=false \
       --vulkan-gpu-workload=individual \
       --osrt-threshold=1000 \
       --resolve-symbols=false \
       --stats=false \
       --force-overwrite=true \
       --output=/tmp/ovrtx-dynamic-gaussian \
       uv run python scripts/profile_ovrtx_dynamic_gaussian.py \
       --asset /mnt/data/isaac_lab_poc/newton_assets/gaussian_splat_toys/baked.BluehairRagdoll_package.usda \
       --fast-simulation \
       --splat-deformation position
   ```

4. In the Nsight timeline, select `OVRTX Dynamic Gaussian Rebuild` and inspect
   its `frame NNN` children. The finite harness contains 30 warm-up frames,
   but this range contains only the 120 steady-state frames. Compare `Newton::step_gpu_sync` to
   `Newton::render`, then expand these NVTX ranges inside the latter:

   * `ViewerRTX::_update_ovrtx_transforms`
   * `ViewerRTX::_update_ovrtx_gaussians`
   * `ViewerRTX::_render_and_display`

5. Optional CPU Tracy capture: launch a Tracy client, then run the harness
   with `NEWTON_RTX_TRACY=1`. OVRTX 0.4.1's released API exposes this through
   `RendererConfig.enable_profiling`; no binary replacement is required.

   ```bash
   rm -f /tmp/newton-gaussian-tracy-go
   NEWTON_RTX_TRACY=1 NEWTON_PROFILE=0 \
       uv run python scripts/profile_ovrtx_dynamic_gaussian.py \
       --asset /mnt/data/isaac_lab_poc/newton_assets/gaussian_splat_toys/baked.BluehairRagdoll_package.usda \
       --fast-simulation \
       --splat-deformation position \
       --wait-for-file /tmp/newton-gaussian-tracy-go
   ```

   Attach the Tracy client while the process waits at the warm state, then run
   `touch /tmp/newton-gaussian-tracy-go` to capture only the steady-state
   interval. The released wheel provides OVRTX's existing CPU zones. The
   deeper Fabric/GPU annotations used during investigation require a Kit build
   and are deliberately not part of this vanilla-wheel repro.

## Control experiment

Add `--no-gaussian-update` to the command. Newton skips both Gaussian skinning
and OVRTX Gaussian streaming/rendering; on the test machine this increases the
fast-simulation run to roughly 30 FPS. This isolates the dynamic Gaussian path
from the simulation-only baseline.
