.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Isaac Lab Integration
=====================

For details about Isaac Lab support for Newton, see the
`Isaac Lab documentation <https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html>`_.


Replicated Deformable Scenes
----------------------------

Large Isaac Lab deformable tasks often replicate many independent worlds that
share the same robot, rigid geometry, and soft body layout.  In finalized
Newton models, ``shape_world`` and ``particle_world`` identify which world owns
each shape and particle; ``-1`` means global.

``CollisionPipeline`` uses that metadata for particle-shape soft contacts.  It
builds a world-aware candidate table and launches contact generation over
``(particle_count, max_shapes_per_world)`` instead of
``particle_count * shape_count``.  Each local particle tests only shapes in its
own world plus global shapes, while global particles test all collidable
shapes.  This keeps replicated scenes linear in the number of worlds and keeps
the launch dimensions suitable for CUDA graph capture.

When a workload needs a fixed soft-contact capacity, pass the same
``soft_contact_max`` to both ``CollisionPipeline`` and ``SolverVBD``.  If the
cap is smaller than the static candidate space, Newton keeps writes bounded and
emits a construction-time warning; contacts beyond the cap are dropped.
