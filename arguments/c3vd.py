"""
Config tuned for C3VD colonoscopy phantom sequences.

Differences vs arguments/endonerf.py:
- Longer training (24000 iters). C3VD scenes are harder than EndoNeRF
  (wide FOV, low texture on phantom mucosa, longer trajectories), and
  PSNR keeps climbing well past the EndoNeRF sweet spot.
- Slightly later densification cutoff + pruning to grow more Gaussians.
- Lower depth_weight because our pseudo-depth comes from Depth-Anything
  which is relative, not metric, and over-weighting it hurts RGB quality.
"""
ModelParams = dict(
    camera_extent=10,
    use_pretrain=False,
)

OptimizationParams = dict(
    coarse_iterations=1500,
    deformation_lr_init=0.00016,
    deformation_lr_final=0.0000016,
    deformation_lr_delay_mult=0.01,
    grid_lr_init=0.0016,
    grid_lr_final=0.000016,
    iterations=24000,
    percent_dense=0.01,
    render_process=True,
    densify_until_iter=4000,
    pruning_from_iter=500,
    densify_from_iter=500,
    densification_interval=100,
    pruning_interval=100,
    opacity_reset_interval=6000,
)

ModelHiddenParams = dict(
    kplanes_config={
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 75],
    },
    multires=[1, 2],
    defor_depth=0,
    net_width=64,
    plane_tv_weight=0.0001,
    time_smoothness_weight=0.01,
    l1_time_planes=0.0001,
    depth_weight=0.005,
    grad_weight=0.001,
    normal_weight=0.001,
    un_img_weight=0.001,
    un_dep_weight=0.001,
    weight_decay_iteration=0,
    bounds=1.6,
    pool_list=[2],
    multi_scale=False,
    use_class_deformation=True,
    semantic_consistency_weight=0.01,
    cutting_threshold=0.1,
    label_update_interval=100,
)

PipelineParams = dict(
    use_depth=True,
    use_smooth=True,
    use_normal=True,
    use_confidence=True,
)
