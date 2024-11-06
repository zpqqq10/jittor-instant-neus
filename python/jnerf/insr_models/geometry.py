import numpy as np

import jittor as jt
from jittor import Module
from jittor import nn
import os

from jnerf import insr_models as insr_models
from jnerf.insr_models.base import BaseModel
from jnerf.insr_models.utils import get_encoding, get_mlp, scale_anything, chunk_batch, contract_to_unisphere
from jnerf.utils.config import get_cfg
from jnerf.utils.common import round_floats
import mcubes

def clamp_(x, vmin, vmax):
    x = jt.stack([
                x[...,0].clamp(vmin[0], vmax[0]),
                x[...,1].clamp(vmin[1], vmax[1]),
                x[...,2].clamp(vmin[2], vmax[2]),
            ], dim=-1)

    return x
            
# marching cube to extract mesh
class MarchingCubeHelper(Module):
    def __init__(self, resolution):
        # super().__init__()
        self.resolution = resolution
        self.points_range = (0, 1)
        self.mc_func = mcubes.marching_cubes
        self.verts = jt.zeros((0,3))

    def grid_vertices(self):
        # if self.verts is None:
        if self.verts.shape[0] == 0:
            x, y, z = jt.linspace(*self.points_range, self.resolution), jt.linspace(*self.points_range, self.resolution), jt.linspace(*self.points_range, self.resolution)
            x, y, z = jt.meshgrid(x, y, z)#indexing='ij'
            verts = jt.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def execute(self, level, threshold=0.):
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
        verts, faces = jt.Var(verts.astype(np.float32)), jt.Var(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }
        

@insr_models.register('volume-sdf')
class VolumeSDF(BaseModel):
    def setup(self):
        self.cfg = get_cfg()
        self.warmup_steps = self.cfg.warm_up_end
        self.n_output_dims = self.config.feature_dim

        per_level_scale = \
            jt.exp((jt.log(self.config.xyz_encoding_config.finest_resolution)-jt.log(self.config.xyz_encoding_config.base_resolution))/(self.config.xyz_encoding_config.n_levels-1))
        self.config.xyz_encoding_config.per_level_scale = float(per_level_scale.numpy()[0])
        # hashgrid encoding
        encoding = insr_models.make(self.config.hash_encoder, self.config.xyz_encoding_config).cuda()
        if self.config.xyz_encoding_config.include_pe:
            self.pe_encoding = get_encoding(3, self.config.xyz_encoding_config).cuda()
            self.pe_encoding.requires_grad_()
            network = get_mlp(encoding.n_output_dims + self.pe_encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config).cuda()
        else:
            network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config).cuda()

        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 1e-3)
        print("geometry finite_difference_eps : ", 'self adjustment' if self.finite_difference_eps == 'progressive' else self.finite_difference_eps)
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None

        self.contraction_type = "AABB"
        self.encoding.requires_grad_()
        self.network.requires_grad_()

        self.iter = 0

        ## set up issurface 
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc']
            self.helper = MarchingCubeHelper(self.config.isosurface.resolution)

        self.bbox_corner = self.cfg.bbox_corner
        self.bbox_size = self.cfg.bbox_size
        self.bbox_radius = self.bbox_size.max() * 0.5 

    def sdf(self, points):
        sdf = self.execute(points, with_grad = False, with_feature = False, with_laplace = False)
        return sdf

    def execute(self, points, with_grad=True, with_feature=True, with_laplace=False):
        points_ = points # points in the original scale
        points = contract_to_unisphere(points, self.bbox_corner, self.bbox_size) # points normalized to (0, 1)
    
        # hash encoding
        PE_points = self.encoding(points.view(-1, 3))
        if self.config.xyz_encoding_config.include_pe:
            # positional encoding
            pe_points = self.pe_encoding(points.view(-1, 3))
            out = self.network(jt.cat([PE_points, pe_points], dim=-1)).view(*points.shape[:-1], self.n_output_dims).float()
        else: 
            out = self.network(PE_points).view(*points.shape[:-1], self.n_output_dims).float()
        sdf, feature = out[...,0], out

        self.iter += 1
        if with_grad:
            if self.grad_type == 'analytic':
                grad = jt.grad(sdf, points_)
            elif self.grad_type == 'finite_difference':
                eps = self._finite_difference_eps

                ### 3 dim size diff
                eps_x = eps * self.bbox_size[0] / self.bbox_size.max()
                eps_y = eps * self.bbox_size[1] / self.bbox_size.max()
                eps_z = eps * self.bbox_size[2] / self.bbox_size.max()
                eps = jt.Var([eps_x, eps_y, eps_z])

                offsets = jt.Var(
                    [
                        [eps_x, 0.0, 0.0],
                        [-eps_x, 0.0, 0.0],
                        [0.0, eps_y, 0.0],
                        [0.0, -eps_y, 0.0],
                        [0.0, 0.0, eps_z],
                        [0.0, 0.0, -eps_z],
                    ]
                ).to(points_)

                points_d_ = clamp_(points_[...,None,:] + offsets, self.bbox_corner, self.bbox_corner + self.bbox_size)
                points_d = contract_to_unisphere(points_d_, self.bbox_corner, self.bbox_size) # points normalized to (0, 1)

                if self.config.xyz_encoding_config.include_pe:
                    # TODO include_pe may cause error here  
                    points_d_sdf = self.network(jt.cat([self.encoding(points_d.view(-1, 3)), self.pe_encoding(points_d.view(-1, 3))], dim=-1))[...,0].view(*points.shape[:-1], 6).float()
                else: 
                    points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                # calculate numerical gradient here 
                grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps  

                if with_laplace:
                    laplace = ((points_d_sdf[..., 0::2] + points_d_sdf[..., 1::2] - 2 * sdf[..., None]) / (eps ** 2))
                
                
        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            assert self.config.grad_type == 'finite_difference', "Laplace computation is only supported with grad_type='finite_difference'"
            rv.append(laplace)


        rv = [v if self.training else v.detach() for v in rv]

        return rv[0] if len(rv) == 1 else rv

    def update_step(self, epoch, global_step):
        self.encoding.update_step(epoch, global_step)
        if self.grad_type == 'finite_difference':
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == 'progressive':
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
                if global_step < self.warmup_steps:
                    current_level = hg_conf.start_level
                else:
                    current_level = min(
                        hg_conf.start_level + 1 + max(global_step - self.warmup_steps, 0) // hg_conf.update_steps,
                        hg_conf.n_levels
                    )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
                grid_size = 2 * self.bbox_radius / grid_res
                if grid_size != self._finite_difference_eps:
                    # rank_zero_info(f"Update finite_difference_eps to {grid_size}")
                    print(f"### Update grid level to {current_level} with mask change.")
                    print(f"### Update finite_difference_eps to {grid_size}")
                # update eps used in finite difference
                self._finite_difference_eps = grid_size
                # update the level mask control of hashgrid encoding
                self.encoding.current_level = current_level
            else:
                raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")

    ### isosurface
    def forward_level(self, points):
        points = contract_to_unisphere(points, self.bbox_corner, self.bbox_size) # points normalized to (0, 1)
        
        if self.config.xyz_encoding_config.include_pe:
            sdf = self.network(jt.cat([self.encoding(points.view(-1, 3)), self.pe_encoding(points.view(-1, 3))], dim=-1)).view(*points.shape[:-1], self.n_output_dims)[...,0].float()
        else: 
            sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0].float()
        
        return sdf

    def isosurface_(self, vmin, vmax):
        def batch_func(x):
            x = jt.stack([
                scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1)
            rv = self.forward_level(x).cpu()
            return rv
    
        level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices())
        mesh = self.helper(level, threshold=self.config.isosurface.threshold)
        mesh['v_pos'] = jt.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)
        return mesh

    def isosurface(self):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse = self.isosurface_(self.bbox_corner, self.bbox_corner + self.bbox_size)
        vmin, vmax = mesh_coarse['v_pos'].min(dim=0), mesh_coarse['v_pos'].max(dim=0)
        vmin_ = clamp_((vmin - (vmax - vmin) * 0.1), self.bbox_corner, self.bbox_corner + self.bbox_size)[0]
        vmax_ = clamp_((vmax + (vmax - vmin) * 0.1), self.bbox_corner, self.bbox_corner + self.bbox_size)[0]
        print('---------------------------')
        print(f'vmin [{round_floats(vmin_[0]):.1f}, {round_floats(vmin_[1]):.1f}, {round_floats(vmin_[2]):.1f}]')
        print(f'vmax [{round_floats(vmax_[0]):.1f}, {round_floats(vmax_[1]):.1f}, {round_floats(vmax_[2]):.1f}]')
        print(f'bbox [{(round_floats(vmax_[0]) - round_floats(vmin_[0])):.1f}, {(round_floats(vmax_[1]) - round_floats(vmin_[1])):.1f}, {(round_floats(vmax_[2]) - round_floats(vmin_[2])):.1f}]')
        print('---------------------------')
        if self.cfg.cal_bbox:
            save_bbox_path = os.path.join(self.cfg.dataset['dataset_dir'], 'bbox.npy')

            bbox_center = ((vmin_ + vmax_) * 0.5).numpy()
            bbox_size = ((vmax - vmin) * 1.5).numpy()

            bbox_info = {
                'bbox_corner': bbox_center - bbox_size * 0.5,
                'bbox_size': bbox_size,
            }
            np.save(save_bbox_path, bbox_info)

        print('coarse pts ',mesh_coarse['v_pos'].shape)
        # extract again for a finer mesh
        mesh_fine = self.isosurface_(vmin_, vmax_)
        print('fine pts ',mesh_fine['v_pos'].shape)
        return mesh_fine
