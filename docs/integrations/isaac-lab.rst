.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Isaac Lab Integration
=====================

For details about Isaac Lab support for Newton, see the
`Isaac Lab documentation <https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html>`_.


Large Isaac Lab Deformable Coupling Repro
-----------------------------------------

The Isaac Lab ``develop`` deformable Kuka/Allegro task exercises a large
replicated scene with a VBD soft body, an MJWarp rigid articulation, Newton's
collision pipeline, and CUDA graph capture.  The following command reproduced
the failures addressed by the Newton changes documented here:

.. code-block:: bash

   ./isaaclab.sh train \
       --rl_library rsl_rl \
       --task Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-v0 \
       --num_envs 1024 \
       --max_iterations 7000 \
       --device cuda:0 \
       physics=fast_two_way

The task config used ``NewtonCollisionPipelineCfg(soft_contact_max=1048576)``
and CUDA graph capture remained enabled.  Before the fixes, two independent
Newton-side issues appeared:

* ``SolverVBD`` preallocated body-particle material buffers as
  ``model.shape_count * model.particle_count`` even when the collision pipeline
  had an explicit ``soft_contact_max``.  With 1024 Isaac Lab environments this
  attempted to allocate a one-dimensional Warp array with
  ``2466349056`` entries and failed before simulation:

  .. code-block:: text

     ValueError: Array shapes must not exceed the maximum representable value
     of a signed 32-bit integer, got 2466349056 in dimension 0.

* After capping the VBD allocation, ``CollisionPipeline.collide()`` still
  launched particle-shape soft contact generation over
  ``particle_count * model.shape_count``.  For the same scene this launched
  ``create_soft_contacts`` with ``dim=2466349056``.  The all-pairs launch is
  both above signed 32-bit indexing and O(worlds^2) for replicated scenes
  because most pairs are later discarded by world-id checks.  Under Warp CUDA
  verification this failed at the soft-contact launch:

  .. code-block:: text

     kernel: create_soft_contacts dim: 2466349056 ...
     Error launching kernel: create_soft_contacts on device cuda:0
     RuntimeError: CUDA error detected: 700

Fixes applied in Newton:

* ``SolverVBD`` accepts an optional ``soft_contact_max`` constructor argument
  and uses it to preallocate body-particle material buffers.  When omitted,
  legacy full candidate sizing is preserved.  This lets graph-captured
  workloads pre-size VBD buffers from the same explicit capacity used by
  ``CollisionPipeline(..., soft_contact_max=...)``.
* ``create_soft_contacts`` passes the explicit soft-contact capacity into
  ``counter_increment`` and only writes capped contact slots.  The
  ``counter_increment`` replay id array write is guarded so a capped
  ``soft_contact_tids`` buffer cannot be indexed by an uncapped launch id.
* ``CollisionPipeline`` now precomputes a per-world particle-shape candidate
  table from ``model.shape_world`` and ``model.particle_world``.  Each world row
  contains local collidable shapes plus global collidable shapes.  Soft contact
  generation uses this table when world metadata exists, launching over
  ``(particle_count, max_shapes_per_world)`` instead of
  ``particle_count * shape_count``.  This keeps the CUDA graph static while
  changing large replicated scenes from O(worlds^2) to O(worlds).

Validation commands run from the Isaac Lab checkout with Newton installed
editable from the local Newton repository:

.. code-block:: bash

   env_isaaclab/bin/python -m pip install 'uv_build>=0.11.0'
   PATH=$PWD/env_isaaclab/bin:$PATH \
       env_isaaclab/bin/python -m pip install --no-build-isolation -e '../newton[sim]'

   ./isaaclab.sh -p source/isaaclab_tasks_experimental/isaaclab_tasks_experimental/manager_based/manipulation/dexsuite_deformable/scripts/diagnose_deformable_task.py \
       --num-envs 1024 \
       --steps 32 \
       --warmup-steps 4 \
       --action-mode random \
       --random-scale 0.25 \
       --physics-preset fast_two_way \
       --device cuda:0 \
       --headless

   ./isaaclab.sh train \
       --rl_library rsl_rl \
       --task Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-v0 \
       --num_envs 1024 \
       --max_iterations 1 \
       --device cuda:0 \
       physics=fast_two_way

Observed validation results:

* 1024-env ``fast_two_way`` diagnostic, CUDA graph enabled: 32 random steps,
  zero resets, zero non-finite observations, finite state, mean step time
  ``22.47 ms``, about ``45562`` env-steps/s.
* 1024-env ``one_way_debug`` diagnostic, CUDA graph enabled and
  ``CUDA_LAUNCH_BLOCKING=1``: 8 random steps, zero resets, zero non-finite
  observations, finite state.
* 1024-env RSL-RL PPO smoke test with ``physics=fast_two_way``:
  graph capture succeeded and iteration 0 collected ``32768`` steps at
  ``14565`` steps/s.
