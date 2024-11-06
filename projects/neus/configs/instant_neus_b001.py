tag = "a smooth stuff, w clean, wo feature decoder dim 8"
dataset_name = 'b_001'

import numpy as np
import math
bbox_corner =  np.array([3, 3, 3])
bbox_size =  np.array([2, 2, 2])

cal_bbox = True

mask_dlt = [51, 2]

# ignore_index= [0, 2, 4, 6, 7, 15, 31, 33, 39, 40, 48, 50, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 126, 127, 151, 170, 171, 174, 178, 179, 180, 181, 185, 186, 187]
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
    n_importance = 128,
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
version = 'Blist'
base_exp_dir = './log/' + version + '/' + dataset_name + '/'
recording = [ './','./models']

dataset = dict(
    type = 'NeuSDataset',
    dataset_dir = '/data/BBB/data/' + dataset_name,
    render_cameras_name = 'cameras_sphere.npz',
    object_cameras_name = 'cameras_sphere.npz',
    
)

if_dilate = False

learning_rate_alpha = 0.05
end_iter = 100000

batch_size = 1024
validate_resolution_level = 4
warm_up_end = 20000
anneal_end = 20000
use_white_bkgd = False

save_freq = 40000
val_freq = 3000
val_num = 2
sample_report_freq = 40000
val_mesh_freq = 10000
report_freq = 2500

grid_resolution = 512
grid_update_freq = 200000

fp16 = False
loss = dict(
  mae_weight = 1.,
  mse_weight = 1.,
  curvature_weight = [end_iter - 10000, 1e-4],  # 5e-4 in neuralangelo
  curv_apply_after = True,
  mask_weight = .1,
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
    feature_dim = 128,
    xyz_encoding_config = dict(
      otype = "ProgressiveBandHashGrid",
      n_levels = 16,
      n_features_per_level = 2,
      log2_hashmap_size = 21,
      base_resolution = 32,
      finest_resolution = 2048,
      include_xyz = True,
      start_step = 0,
      start_level = 2,
      update_steps = 5000,
      ### !
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
      ### sdf layer ? 
      n_hidden_layers=1,
      sphere_init=True,
      sphere_init_radius=0.5,
      weight_norm=False,
    ),
    isosurface = dict(
      method = 'mc',
      resolution = 512,
      chunk = 2097152 // 512,
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

    nerf_pos_encoder = dict(
        type='FrequencyEncoder',
        multires=6,
        input_dims=3,
    ),

    ### feature decoder
    mlp_network_config = dict(
      otype = "VanillaMLP",
      activation = "none",
      output_activation = "none",
      n_neurons = 64,
      n_hidden_layers = 2,
    ),
    # color_activation = "sigmoid",
    n_feature = 3,
)


export = dict(
  chunk_size = 2097152 // 512,
  export_vertex_color = False,
  mask_grid_pruning = True
)
