Add `newton/examples/multiphysics/example_mujoco_vbd_gaussian_twin.py`, a
MuJoCo/VBD coupled example that simulates a 3D Gaussian splat asset as a soft
body: the asset's tetrahedral mesh is imported through `ModelBuilder.add_usd`
and simulated with VBD, while the render-only Gaussian field is deformed every
frame from the skinning weights baked into the asset. The example ships two
scenes: `grasp`, where a proxy-coupled gripper picks the toy up off the ground,
and `sway`, which swings it once it is held. Supply the large third-party USD
package with `--asset` (or `NEWTON_GAUSSIAN_TWIN_ASSET`).

Use `--splat-deformation` to select center-only, center-and-rotation, or
center-and-rotation-and-scale visualization deformation. The scale mode uses
the local deformation-gradient axis lengths, so it captures axial stretch but
does not represent shear exactly.
