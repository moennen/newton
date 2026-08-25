Add `newton/examples/multiphysics/example_mujoco_vbd_gaussian_twin.py`, a
MuJoCo/VBD coupled example that simulates a 3D Gaussian splat asset as a soft
body: a tetrahedral cage is voxelized from the Gaussian centers, VBD simulates
the cage, and the render-only Gaussian field is skinned to it every frame.
