import jittor as jt
import jittor.nn as nn

import numpy as np
import os

import jnerf.insr_models as insr_models
from jnerf.insr_models.ray_aabb import Ray_AABB

from jnerf.utils.config import get_cfg
from jnerf.utils.registry import SAMPLERS
from jnerf.insr_models.utils import chunk_batch, get_mlp
from jnerf.models.samplers.neus_render.renderer import extract_geometry
from tqdm import tqdm


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = jt.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds - 1), inds - 1)
    above = jt.minimum((cdf.shape[-1] - 1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)


    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = jt.where(denom < 1e-5, jt.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.variance = jt.Var(init_val)

    def execute(self, x):
        return jt.ones([len(x), 1]) * jt.exp(self.variance * 10.0)

def save_obj(pts, path):

    obj_path = path
    obj_file = open(obj_path, 'w')
    N = pts.shape[0]
    num_sample = pts.shape[1]

    c0 = jt.Var([1., 0., 0.])
    c1 = jt.Var([0., 1., 0.])

    for i in tqdm(range(N)):
        for j in range(num_sample):

            c_R = (j/num_sample) * c1 + (1.0 - j/num_sample) * c0
            
            obj_file.write("v {0} {1} {2} {3} {4} {5}\n".format(pts[i,j,0],pts[i,j,1],pts[i,j,2], c_R[0], c_R[1], c_R[2]))
    
    obj_file.close()

@SAMPLERS.register_module()
class InstantNeuSRenderer(nn.Module):
    def __init__(self,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        
        self.cfg = get_cfg()
        
        self.nerf = None
        self.sdf_network = None
        self.deviation_network = None
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside

        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

        # sdf
        self.geometry = insr_models.make(self.cfg.geometry.name, self.cfg.geometry)
        # mlp to decode the feature to rgb
        self.texture = insr_models.make(self.cfg.texture.name, self.cfg.texture)
        self.deviation_network = SingleVarianceNetwork(self.cfg.model.variance_network.init_val)

        if self.cfg.texture.feature_decoder_config:
            self.rgb_feature_decoder = get_mlp(self.cfg.texture.n_feature, 3, self.cfg.texture.feature_decoder_config) 

        self.geometry.requires_grad_()
        self.texture.requires_grad_()
        self.deviation_network.requires_grad_()
        self.iter = 0

        self.samples_path = None

        ## mask grid 
        self.Grid = None

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = jt.norm(pts, p=2, dim=-1, keepdim=False, eps=1e-6)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = jt.concat([jt.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = jt.stack([prev_cos_val, cos_val], dim=-1)
        cos_val = jt.min(cos_val, dim=-1, keepdims=False)
        cos_val = cos_val.safe_clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = jt.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = jt.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * jt.cumprod(
            jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-6], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):

        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = jt.concat([z_vals, new_z_vals], dim=-1)
        index,z_vals = jt.argsort(z_vals, dim=-1)

        if not last:
            new_sdf = self.geometry(pts.reshape(-1, 3), with_grad=False, with_feature=False).reshape(batch_size, n_importance)
            sdf = sdf.reshape(batch_size, n_samples)
            sdf = jt.concat([sdf, new_sdf], dim=-1)
            xx = jt.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jt.concat([dists, jt.Var([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3

        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        if self.cfg.geometry.grad_type == "finite_difference": 
            # numerical gradient
            sdf, gradients, feature_vector, sdf_laplace = sdf_network(pts, with_laplace=True)
        else: 
            sdf, gradients, feature_vector = sdf_network(pts)

        normal = jt.normalize(gradients, p=2, dim=-1)
        # for ln loss
        ln = sdf_laplace.mul(normal).sum(-1)
        sampled_color_feature = color_network(feature_vector, dirs, normal, pts).reshape(batch_size, n_samples, -1)

        # weights for volume rendering
        inv_s = deviation_network(jt.zeros([1, 3]))[:, :1].safe_clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        true_cos = (dirs * normal).sum(-1, keepdims=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(nn.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     nn.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf.reshape(-1, 1) + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf.reshape(-1, 1) - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = jt.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = jt.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).safe_clip(0.0, 1.0)
        weights = alpha * jt.cumprod(jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-6], -1), -1)[:, :-1]

        # do not render with background
        if background_alpha is not None:
            pass

        # for debug, check out the sample rays
        if self.iter % self.cfg.sample_report_freq == 0:
            sdf_ = sdf.reshape(batch_size, n_samples)[:30]
            pts_ = pts.reshape(batch_size, n_samples, 3)[:30]
            weight_ = weights.reshape(batch_size, n_samples)[:30]
            print("save obj at: ",os.path.join(self.samples_path, str(self.iter) + '.obj'))

            save_obj(pts_.numpy(), os.path.join(self.samples_path, str(self.iter) + '.obj'))
            np.savetxt(os.path.join(self.samples_path, str(self.iter) + '_sdf.txt'), sdf_, fmt='%.5f')
            np.savetxt(os.path.join(self.samples_path, str(self.iter) + '_weight.txt'), weight_, fmt='%.5f')
            
        weights_sum = weights.sum(dim=-1, keepdims=True)
        opacity = weights_sum
        color_feature = (sampled_color_feature * weights[:, :, None]).sum(dim=1)

        ### using rgb feature decoder
        if self.cfg.texture.feature_decoder_config:
            color = self.rgb_feature_decoder(color_feature)
        ### direct output rgb img
        else:
            color = color_feature

        normal = normal.reshape(batch_size, n_samples, 3)
        comp_normal = (normal * weights[:, :, None]).sum(dim=1)
        comp_normal = jt.normalize(comp_normal, p=2, dim=-1)
        depth = jt.mul(mid_z_vals, weights).sum(dim=1)
        comp_rgb = color

        render_out = {
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            # 'num_samples': jt.Var([n_samples], dtype=jt.int32),
            'sdf_samples': sdf,
            'sdf_grad_samples': gradients,
            'weights': weights.view(-1),
            'pts': pts.view(-1, 3),
            'inv_s' : inv_s,
            'ln' : ln,
        }
        
        if self.cfg.geometry.grad_type == "finite_difference": 
            render_out.update({
                'sdf_laplace_samples': sdf_laplace.sum(-1),
            })

        return render_out
        

    def render(self, rays_o, rays_d, near, far, background_rgb=None, cos_anneal_ratio=0.0):
        valid_mask = jt.ones(rays_o.shape[0])
        valid_mask = (valid_mask == 1).nonzero()[:, 0]
        if self.cfg.cal_bbox == True:
            bbox_corner = jt.Var(self.cfg.bbox_corner)
            bbox_size = jt.Var(self.cfg.bbox_size)
            bbox_center = bbox_corner + bbox_size * 0.5
            bounds = Ray_AABB.apply(rays_o, rays_d, bbox_center, bbox_size)
            _valid_mask = (bounds[:, 0] != -1).nonzero()[:, 0]
            
            valid_sum = _valid_mask.shape[0]
            if valid_sum != 0:
                valid_mask = _valid_mask
                near =  bounds[valid_mask, :1]
                far = bounds[valid_mask, -1:]
                rays_o = rays_o[valid_mask]
                rays_d = rays_d[valid_mask]   
            else: 
                pass

        sample_dist = self.cfg.bbox_size.max() / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = jt.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        # if there is an occupancy grid, determine valid_mask with the grid
        if self.Grid != None:
            occupancy_grid = self.Grid
            new_z_vals = occupancy_grid.sample_z_vals(rays_o, rays_d, self.n_samples)
            pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]  # batch_size, n_samples, 3
            sum_z = new_z_vals.sum(dim = 1)
            valid_mask = (sum_z > 0.).nonzero()[:, 0]
            valid_sum = valid_mask.shape[0]

            if valid_sum == 0:
                return None
                
            z_vals = new_z_vals[valid_mask]
            rays_o = rays_o[valid_mask]
            rays_d = rays_d[valid_mask]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = jt.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        if self.n_outside > 0:
            z_vals_outside = far / jt.flip(z_vals_outside, dim=-1) + 1.0 / self.n_samples

        # Up sample
        if self.n_importance > 0:
            with jt.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.geometry(pts.reshape(-1, 3), with_grad=False, with_feature=False)#  sdf_grad, feature 
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        background_alpha = None
        background_sampled_color = None
        # Background model
        if self.n_outside > 0:
            z_vals_feed = jt.concat([z_vals, z_vals_outside], dim=-1)
            _, z_vals_feed = jt.argsort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        render_out = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.geometry,
                                    self.deviation_network,
                                    self.texture,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        if self.Grid != None:
            self.Grid.update_grid_weight(self.iter, render_out['pts'], render_out['weights'])

        render_out['valid_mask'] = valid_mask
        return render_out

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.geometry.sdf(pts)
                                # query_func=lambda pts: -self.sdf_network.sdf(pts)
                                )

    
    def update_step(self, epoch, global_step):
        self.iter = global_step
        self.geometry.update_step(epoch, global_step)
        self.texture.update_step(epoch, global_step)
    
    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def export(self, export_config):
        mesh = self.isosurface()

        ### clean outside pts:
        if export_config.mask_grid_pruning and not self.cfg.cal_bbox:
            # use occupancy grid to prune invalid points
            pts_mask = self.Grid.points_mask(mesh['v_pos'])
            print("mesh faces: ", mesh['t_pos_idx'].shape)

            mask = pts_mask.int()#.nonzero()[..., 0]
            sum = jt.cumsum(mask)
            sum = sum - 1

            f = mesh['t_pos_idx']
            M = f.shape[0]
            new_f = jt.zeros((M, 3))
            num_f = 0
            for i in tqdm(range(f.shape[0])):
                vert = f[i]
                x, y, z = vert
                if mask[x] and mask[y] and mask[z]:
                    new_f[num_f] = jt.Var([sum[x], sum[y], sum[z]])[..., 0]
                    num_f += 1
            new_f = new_f[:num_f]

            print('new faces num: ', new_f.shape)
            vert = mesh['v_pos']
            left = pts_mask.nonzero()[..., 0]
            new_v = vert[left]
            print('new verts num: ', new_v.shape)

            mesh['v_pos'] = new_v
            mesh['t_pos_idx'] = new_f

        if export_config.export_vertex_color:
            print("shape of mesh pts: ",mesh['v_pos'].shape)
            _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'], with_grad=True, with_feature=True)
            normal = jt.normalize(sdf_grad, p=2, dim=-1)
            rgb = self.texture(feature, -normal, normal, mesh['v_pos']) # set the viewing directions to the normal to get "albedo"
            rgb = rgb[:, [2,1,0]]
            mesh['v_rgb'] = rgb.cpu()
        return mesh
