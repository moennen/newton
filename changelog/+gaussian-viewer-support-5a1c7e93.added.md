Add Gaussian splat rendering to `ViewerUSD` and `ViewerRTX`: `log_gaussian` now
authors a `ParticleField3DGaussianSplat` prim and streams the per-frame splat
centers and orientations, so a deforming Gaussian field can be recorded to USD or
ray traced. Previously only `ViewerGL` and `ViewerViser` drew Gaussians.
