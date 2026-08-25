Add `newton/examples/multiphysics/example_mujoco_vbd_gaussian_twin.py`, a
MuJoCo/VBD coupled example that simulates a 3D Gaussian splat asset as a soft
body: the asset's tetrahedral mesh is imported through `ModelBuilder.add_usd`
and simulated with VBD, while the render-only Gaussian field is deformed every
frame from the skinning weights baked into the asset. The example ships two
scenes: `grasp`, where a proxy-coupled gripper picks the toy up off the ground,
and `sway`, which swings it once it is held.
