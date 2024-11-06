tag = "dev handbag wo curv"
dataset_name = 'handbag_019'

import numpy as np
import math
bbox_corner = np.array([-0.3, -0.4, -0.2])
bbox_size = np.array([0.6, 0.6, 0.6])
cal_bbox = False
# bbox_corner = np.array([-1, -1, -1])
# bbox_size = np.array([2, 2, 2])
# cal_bbox = True
# dilation parameters
mask_dlt = [41, 1]
# which images to discard
ignore_index= []

model = dict(
    type = 'NeuS',
    variance_network = dict(
        init_val = 0.3,
    ),
)

render = dict(
    type = 'InstantNeuSRenderer',
    n_samples = 512,
    n_importance = 0,
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

# version = 'instant_3.0_v'
version = 'less_sample'
base_exp_dir = './log/' + version + '/' + dataset_name + '/'
recording = [ './','./models']

dataset = dict(
    type = 'NeuSDataset',
    dataset_dir = '/data/AAA/data/' + dataset_name,
    render_cameras_name = 'cameras_sphere.npz',
    object_cameras_name = 'cameras_sphere.npz',
)

# whether to dilate the mask
if_dilate = True

learning_rate_alpha = 0.05
end_iter = 80000

batch_size = 1024
# to determine a lower resolution for fast validation
validate_resolution_level = 6
warm_up_end = 10000
anneal_end = 20000
use_white_bkgd = False

# when to save checkpoints
save_freq = 10000
# validate images with rgb, normal and depth
val_freq = 5000
# how many images to validate
val_num = 3
# when to visualize the sample points for debug
sample_report_freq = 40000
# when to export meshes
val_mesh_freq = 10000
# when to write logs with tensorboard
report_freq = 2000

grid_resolution = 512
grid_update_freq = 5000

fp16 = False
loss = dict(
  mae_weight = 1.,
  mse_weight = 1.,
  curvature_weight = [end_iter - 10000, 1e-5],  # 5e-4 in neuralangelo
  curv_apply_after = True,
  mask_weight = 1.,
  eikonal_weight = .1,
  opaque_weight = [20000, .1],  # .1 only apply opaque loss after 50,000 iterations
)

geometry = dict(
    name = "volume-sdf",
    hash_encoder = 'HashGrid',
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
      per_level_scale = 1.3195079107728942,
      include_xyz = True,
      start_step = 0,
      start_level = 4,
      update_steps = 2000,
      include_pe = False,
      pos_encoder = dict(
        type='FrequencyEncoder',
        multires=6,
        input_dims=3,
    ),
    ),
    finite_difference_eps = 'progressive',
    mlp_network_config = dict(
      otype="VanillaMLP",
      activation="ReLU",
      output_activation="none",
      n_neurons=64,
      n_hidden_layers=1,
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

hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
hash_grow_rate = math.pow(geometry['xyz_encoding_config']['finest_resolution'] / geometry['xyz_encoding_config']['base_resolution'], 1. / (geometry['xyz_encoding_config']['n_levels'] - 1))
print('hash grow rate: ',hash_grow_rate)

texture = dict(
    name = "volume-radiance",
    input_feature_dim = geometry['feature_dim'] + 3, # surface normal as additional input
    dir_encoding_config = dict(
      type = "SHEncoder",
      degree = 4,
    ),
    # nerf_pos_encoder = dict(
    #     type='FrequencyEncoder',
    #     multires=6,
    #     input_dims=3,
    # ),
    mlp_network_config = dict(
      otype = "VanillaMLP",
      activation = "ReLU",
      output_activation = "none",
      n_neurons = 64,
      n_hidden_layers = 2,
    ),
    ### feature decoder
    n_feature = 3,
    color_activation = "sigmoid",
)


export = dict(
  chunk_size = 2097152 // 16,
  export_vertex_color = True,
  mask_grid_pruning = True
)
