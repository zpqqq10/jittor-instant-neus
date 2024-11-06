# tag = "15-bit feature, mse + l1 + 1e-4 curvature + opaque loss, 2000 update step & 8-bit feature for progressive, 30000 iterations"
dataset_name = 'ornaments_005'

model = dict(
    type = 'NeuS',
    variance_network = dict(
        init_val = 0.3,
    ),
)

render = dict(
    type = 'InstantNeuSRenderer',
    n_samples = 256,
    n_importance = 64,
    n_outside = 0,           # w/o outside nerf to render background
    up_sample_steps = 4,     # 1 for simple coarse-to-fine sampling
    perturb = 0,
)



optim = dict(
    type='Adam',
    lr=1e-2,
    eps=1e-15,
    betas=(0.9,0.99), 
)


version = 'instant_3.0_v'
base_exp_dir = './log/' + version + '/' + dataset_name + '/'
recording = [ './','./models']

dataset = dict(
    type = 'NeuSDataset',
    dataset_dir = '/home/zhangxin/Documents/DATA/contest/' + dataset_name,
    render_cameras_name = 'cameras_sphere.npz',
    object_cameras_name = 'cameras_sphere.npz',
    
)

learning_rate_alpha = 0.05
end_iter = 30000

batch_size = 512
validate_resolution_level = 5
warm_up_end = 200
anneal_end = 20000
use_white_bkgd = False

save_freq = 10000
val_freq = 2000
val_mesh_freq = 2000
report_freq = 1000

import numpy as np
bbox_corner = np.array([0.0, -0.4, -0.3])
bbox_size = np.array([0.3, 0.3, 0.35])

hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"

fp16 = False
loss = dict(
  mae_weight = 1.,
  mse_weight = 1.,
  curvature_weight = 5e-4,  # 5e-4 in neuralangelo
  mask_weight = .1,
  eikonal_weight = .1,
  opaque_weight = [20000, .1],  # .1 only apply opaque loss after 50,000 iterations
)

geometry = dict(
    name = "volume-sdf",
    hash_encoder = 'HashGrid',
    pos_encoder = dict(
    type='HashEncoder',
    ),
    radius = 1.0,
    # grad_type = "analytic",
    grad_type = "finite_difference",
    aabb_scale = 1,
    feature_dim = 64,
    xyz_encoding_config = dict(
      otype = "ProgressiveBandHashGrid",
      n_levels = 16,
      n_features_per_level = 2,
      log2_hashmap_size = 19,
      base_resolution = 32,
      finest_resolution = 2048,
      include_xyz = True,
      start_step = 0,
      start_level = 4,
      update_steps = 1500,
    ),
    finite_difference_eps = 'progressive',
    mlp_network_config = dict(
      otype="VanillaMLP",
      activation="ReLU",
      output_activation="none",
      n_neurons=64,
      n_hidden_layers=2,
      sphere_init=True,
      sphere_init_radius=0.5,
      weight_norm=False,
    ),
    isosurface = dict(
      method = 'mc',
      resolution = 512,
      chunk = 2097152 // 16,
      threshold = 0.,
    ),
)

texture = dict(
    name = "volume-radiance",
    input_feature_dim = geometry['feature_dim'] + 3, # surface normal as additional input
    dir_encoding_config = dict(
      type = "SHEncoder",
      degree = 4,
    ),
    mlp_network_config = dict(
      otype = "VanillaMLP",
      activation = "ReLU",
      output_activation = "none",
      n_neurons = 64,
      n_hidden_layers = 2,
    ),
    color_activation = "sigmoid",
)


export = dict(
  chunk_size = 2097152 // 32,
  export_vertex_color = True,
)

