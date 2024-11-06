
import math
import jittor as jt
from jittor import Module
from jittor import nn

from jnerf import insr_models as insr_models
from jnerf.insr_models.utils import get_activation, get_encoding, get_mlp, scale_anything, contract_to_unisphere
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS
from jnerf.utils.config import get_cfg

@insr_models.register('volume-radiance')
class VolumeRadiance(Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.cfg = get_cfg()

        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = self.config.n_feature

        self.bbox_size = self.cfg.bbox_size
        self.bbox_corner = self.cfg.bbox_corner
        
        # direction encoding
        encoding = get_encoding(self.n_dir_dims, self.config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims

        if self.config.nerf_pos_encoder:
            self.embed_fn = build_from_cfg(self.config.nerf_pos_encoder, ENCODERS)
            self.input_ch = self.embed_fn.out_dim
            print("input_ch: ",self.input_ch,  self.n_input_dims)
            self.n_input_dims += self.input_ch 

        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config) 
        
        self.encoding = encoding
        self.network = network
        self.iter = 0

    def execute(self, features, dirs, normal, pts):

        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))

        if self.config.nerf_pos_encoder:
            pts_embd = self.embed_fn(contract_to_unisphere(pts, self.bbox_corner, self.bbox_size))
            network_inp = jt.cat([features.view(-1, features.shape[-1]), dirs_embd, normal.view(-1, normal.shape[-1]), pts_embd], dim=-1)
        else:
            network_inp = jt.cat([features.view(-1, features.shape[-1]), dirs_embd, normal.view(-1, normal.shape[-1])], dim=-1)
        
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        self.iter = global_step

    def regularizations(self, out):
        return {}

