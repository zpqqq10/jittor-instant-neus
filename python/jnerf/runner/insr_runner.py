import os
import time
import logging
import argparse
import numpy as np
import datetime
import cv2 as cv
import trimesh
from tqdm import tqdm
from shutil import copyfile
from jittor.dataset import Dataset

from jnerf.utils.config import init_cfg, get_cfg
from jnerf.utils.registry import build_from_cfg,NETWORKS,SCHEDULERS,DATASETS,OPTIMS,SAMPLERS,LOSSES

import jittor as jt
import jittor.nn as nn
# for log
from tensorboardX import SummaryWriter
from jnerf.insr_models.occupancy_grid import Occupancy_Grid

class InstantNeuSRunner:
    def __init__(self, mode='train', is_continue=False, cfg_path=''):
        # Configuration
        self.cfg = get_cfg()

        # basic
        current = np.datetime64(datetime.datetime.now())
        current_time = str(current).split('.')[0]

        self.base_exp_dir = os.path.join(self.cfg.base_exp_dir, current_time)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.iter_step = 0
        # backup config file
        copyfile(cfg_path, os.path.join(self.base_exp_dir, f'{current_time}.py'))
        
        # tensorboard writer
        self.writer = SummaryWriter(logdir=os.path.join(self.base_exp_dir, 'logs'))

        # Training parameters
        self.end_iter = self.cfg.end_iter
        # when to save checkpoints
        self.save_freq = self.cfg.save_freq
        # when to write logs with tensorboard
        self.report_freq = self.cfg.report_freq
        # validate images with rgb, normal and depth
        self.val_freq = self.cfg.val_freq
        # how many images to validate
        self.val_num = self.cfg.val_num
        # when to export meshes
        self.val_mesh_freq = self.cfg.val_mesh_freq
        self.batch_size = self.cfg.batch_size
        # to determine a lower resolution for fast validation
        self.validate_resolution_level = self.cfg.validate_resolution_level
        self.learning_rate_alpha = self.cfg.learning_rate_alpha
        self.use_white_bkgd = self.cfg.use_white_bkgd
        self.warm_up_end = self.cfg.warm_up_end
        self.anneal_end = self.cfg.anneal_end

        self.is_continue = is_continue
        self.dataset      = build_from_cfg(self.cfg.dataset, DATASETS)
        self.renderer     = build_from_cfg(self.cfg.render, SAMPLERS)
        # path to save samples for debug
        self.renderer.samples_path = os.path.join(self.base_exp_dir, 'samples')
        os.makedirs(os.path.join(self.base_exp_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
    
        self.n_images = self.dataset.n_images
        self.learning_rate = self.cfg.optim.lr
        self.optimizer = build_from_cfg(self.cfg.optim, OPTIMS, params=self.renderer.parameters())

        self.Grid = \
        Occupancy_Grid(self.cfg.bbox_corner, self.cfg.bbox_size, self.cfg.grid_resolution)
    
        self.load_tag = False
        self.update_steps = self.cfg.geometry.xyz_encoding_config.update_steps

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            assert self.cfg.load_path is not None and self.cfg.load_path != '', 'Please specify the load path in the config file.'
            load_path = self.cfg.load_path

            # load the latest checkpoint
            model_list_raw = os.listdir(os.path.join(load_path, 'checkpoints'))
            model_list = []
            grid_list = []
            grid_weight_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pkl' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
                if model_name[-9:] == '_grid.npy' and int(model_name[:-9]) <= self.end_iter:
                    grid_list.append(model_name)
                if model_name[-16:] == '_grid_weight.npy' and int(model_name[:-16]) <= self.end_iter:
                    grid_weight_list.append(model_name)
            model_list.sort()
            grid_list.sort()
            grid_weight_list.sort()
            latest_model_name = model_list[-1]
            if len(grid_list): 
                latest_grid_name = grid_list[-1]
            else:
                latest_grid_name = None
            if len(grid_weight_list):
                latest_grid_weight_name = grid_weight_list[-1]
            else: 
                latest_grid_weight_name = None

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name, latest_grid_name, latest_grid_weight_name, load_path)
        jt.set_global_seed(42)

    def binary_cross_entropy(self, input, target):
        """
        F.binary_cross_entropy is not numerically stable in mixed-precision training.
        """
        return -(target * jt.log(input) + (1 - target) * jt.log(1 - input)).mean()

    # prune uninterseting vertices according to the rays within masks
    # visual hull
    def grid_pruning_init(self):
        self.Grid.counts_init()

        save_occupancy_path = os.path.join(self.cfg.dataset.dataset_dir, 'occupancy_' + str(self.cfg.grid_resolution) + '.npy')
        if os.path.exists(save_occupancy_path):
            self.Grid.load(save_occupancy_path)
        else:
            print('pruning mask with input img.')
            for idx in tqdm(range(self.dataset.n_images)):

                rays_o, rays_d = self.dataset.gen_rays_at_wmask(idx)

                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                
                self.Grid.grid_pruning(rays_o, rays_d)
            
            self.Grid.occupied_grid_init()
            self.Grid.save(os.path.join(self.cfg.dataset.dataset_dir, 'occupancy_' + str(self.cfg.grid_resolution) + '.npy'), )

    def train(self):
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()


        if self.cfg.cal_bbox:
            print("cal bbox, not using grid")
        else:
            if self.load_tag == False:
                self.grid_pruning_init()
            self.renderer.Grid = self.Grid

        self.if_finest = False
        self.finest_step = -1
        self.n_levels = self.cfg.geometry['xyz_encoding_config']['n_levels']
        self.start_level = self.cfg.geometry['xyz_encoding_config']['start_level']

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.get_batch_random_rays(self.batch_size, apply_mask = True)
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]

            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = jt.ones([1, 3])

            fg_mask = mask.squeeze()
            self.renderer.update_step(0, iter_i)
            out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())
            
            #######************ loss calculation ************########
            valid_mask = out['valid_mask']
            assert(valid_mask.shape[0] == out['comp_rgb'].shape[0])
            true_rgb = true_rgb[valid_mask]
            fg_mask = fg_mask[valid_mask]

            loss_rgb_l1 = nn.l1_loss(out['comp_rgb'][out['rays_valid'][...,0]], true_rgb[out['rays_valid'][...,0]])
            loss_rgb_l2 = nn.mse_loss(out['comp_rgb'][out['rays_valid'][...,0]], true_rgb[out['rays_valid'][...,0]])
            loss_eikonal = ((jt.norm(out['sdf_grad_samples'], dim=-1) - 1.)**2).mean()
            
            opacity = jt.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
            loss_mask = self.binary_cross_entropy(opacity, fg_mask.float())
            
            inv_s = out['inv_s']
            inv_s = inv_s.mean()[0].numpy()
            loss = loss_rgb_l1 * self.cfg.loss.mae_weight + \
                   loss_rgb_l2 * self.cfg.loss.mse_weight + \
                   loss_eikonal * self.cfg.loss.eikonal_weight + \
                   loss_mask * self.cfg.loss.mask_weight

            # if self.iter_step > self.cfg.loss.opaque_weight[0]:
            #     loss_opaque = self.binary_cross_entropy(opacity, opacity)
            #     loss += loss_opaque * self.cfg.loss.opaque_weight[1]
            
            loss_curvature = 0.
            if type(self.cfg.loss.curvature_weight) == list: 
                
                assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
                loss_curvature = out['sdf_laplace_samples'].abs().mean()
                # if self.cfg.loss.curv_apply_after is True and self.cfg.loss.curvature_weight[0] < self.iter_step:
                #     loss += loss_curvature * self.cfg.loss.curvature_weight[1]
                # elif self.cfg.loss.curv_apply_after is False and self.cfg.loss.curvature_weight[0] > self.iter_step:
                #     loss += loss_curvature * self.cfg.loss.curvature_weight[1]
                # else: 
                #     loss += loss_curvature * 0

                # if self.iter_step < self.update_steps:
                #     curvature_weight = self.iter_step / self.update_steps * self.cfg.loss.curvature_weight[1]
                # else:
                #     decay_factor = self.cfg.hash_grow_rate ** ((self.iter_step - self.update_steps) / self.update_steps)
                #     curvature_weight = self.cfg.loss.curvature_weight[1] / decay_factor

                if self.iter_step < self.warm_up_end:
                    curvature_weight = self.iter_step / self.warm_up_end * self.cfg.loss.curvature_weight[1]
                else:
                    decay_factor = self.cfg.hash_grow_rate ** ((self.iter_step - self.warm_up_end) / self.update_steps)
                    curvature_weight = self.cfg.loss.curvature_weight[1] / decay_factor

                loss += loss_curvature * curvature_weight

            elif type(self.cfg.loss.curvature_weight) == float:
                assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
                loss_curvature = out['sdf_laplace_samples'].abs().mean()
                loss += loss_curvature * self.cfg.loss.curvature_weight
            
            if self.cfg.loss.ln_weight:
                ln_weight = self.cfg.loss.ln_weight
                ln_weight *= min(0.5 * self.iter_step / self.warm_up_end, 1.)
                loss_ln = out['ln'].abs().mean()
                loss += ln_weight * loss_ln

            #######************ loss calculation end ************########

            self.iter_step += 1

            if self.iter_step % 100 == 0:
                assert jt.misc.isfinite(loss_rgb_l1), f"Loss l1 not finite in {self.iter_step}!"
                assert jt.misc.isfinite(loss_rgb_l2), f"Loss l2 not finite in {self.iter_step}!"
                assert jt.misc.isfinite(loss_eikonal), f"Loss eikonal not finite in {self.iter_step}!"
                assert jt.misc.isfinite(loss_mask), f"Loss mask not finite in {self.iter_step}!"
                assert jt.misc.isfinite(loss_curvature), f"Loss curv not finite in {self.iter_step}!"
                self.writer.add_scalar('train/l1_loss', loss_rgb_l1.numpy(), self.iter_step)
                self.writer.add_scalar('train/l2_loss', loss_rgb_l2.numpy(), self.iter_step)
                self.writer.add_scalar('train/eikonal_loss', loss_eikonal.numpy(), self.iter_step)
                self.writer.add_scalar('train/mask_loss', loss_mask.numpy(), self.iter_step)
                if self.cfg.loss.curvature_weight: 
                    self.writer.add_scalar('train/curvature_loss', loss_curvature.numpy(), self.iter_step)
                    self.writer.add_scalar('train/curvature_weight', curvature_weight, self.iter_step)
                if self.cfg.loss.ln_weight: 
                    self.writer.add_scalar('train/loss_ln', loss_ln.numpy(), self.iter_step)
                self.writer.add_scalar('train/inv_s', inv_s, self.iter_step)
                self.writer.add_scalar('train/loss', loss.numpy(), self.iter_step)
                for name, param in self.renderer.geometry.network.layers.named_parameters():
                    assert jt.misc.isfinite(param).any(), f"Param {name} not finite in {self.iter_step}!"
                for name, param in self.renderer.texture.network.layers.named_parameters():
                    assert jt.misc.isfinite(param).any(), f"Param {name} not finite in {self.iter_step}!"
                
            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            self.optimizer.step()
            
            if self.iter_step % self.report_freq == 0:
                for name, param in self.renderer.geometry.network.layers.named_parameters():
                    gr = param.opt_grad(self.optimizer)
                    gr = gr.numpy()
                    self.writer.add_histogram('gradient/geometry'+name, gr, self.iter_step)
                for name, param in self.renderer.texture.network.layers.named_parameters():
                    gr = param.opt_grad(self.optimizer)
                    gr = gr.numpy()
                    self.writer.add_histogram('gradient/texture'+name, gr, self.iter_step)

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                print(f'---- validate with resolution {self.validate_resolution_level} of first {self.cfg.val_num} images ----')
                for ttt in tqdm(range(self.cfg.val_num)):
                    self.validate_image(resolution_level = self.validate_resolution_level)

            if self.iter_step % self.val_mesh_freq == 0:
                self.export()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

            if self.if_finest and (self.iter_step - self.finest_step) % self.cfg.grid_update_freq == 0:
                self.Grid.update_occupancy(self.iter_step)
                self.Grid.save(os.path.join(self.base_exp_dir, 'checkpoints', str(self.iter_step) + '_grid.npy'), os.path.join(self.base_exp_dir, 'checkpoints', str(self.iter_step) + '_grid_weight.npy'))

            if self.renderer.geometry.encoding.current_level >= self.start_level + 3 and self.if_finest == False:
                self.finest_step = self.iter_step
                self.Grid.clean_weight()
                print(' --- Finest level at iter ',self.iter_step, ' clean grid weight for prune. ---')
                self.if_finest = True
        
        self.writer.close()
        self.export()


        self.save_checkpoint()
        ## end train
        vals = self.n_images // 50
        print(f'---- validate with full resolution of first {vals} images ----')
        ## randomly select 1/4 to validate
        for val_idx in tqdm(range(vals)):
            self.validate_image(resolution_level = 1)

    def get_image_perm(self):
        return jt.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def load_checkpoint(self, checkpoint_name, grid_name, grid_weight_name, load_path):
        checkpoint_path = os.path.join(load_path, 'checkpoints')
        print(f'---load {checkpoint_path}/{checkpoint_name} of {self.cfg.dataset_name}---')

        self.load_tag = True

        checkpoint = jt.load(os.path.join(checkpoint_path, checkpoint_name))
        self.renderer.load_state_dict(checkpoint['insr'])
        self.iter_step = checkpoint['iter_step']
        self.renderer.update_step(0, self.iter_step)
        if grid_name is not None:
            print(f'---load {checkpoint_path}/{grid_name} and {checkpoint_path}/{grid_weight_name} of {self.cfg.dataset_name}---')
            self.Grid.load(os.path.join(checkpoint_path, grid_name), os.path.join(checkpoint_path, grid_weight_name))

        print('grid loaded, valid grid num : ', self.Grid.occupancy_grid.sum())


    def save_checkpoint(self):
        checkpoint = {
            'insr': self.renderer.state_dict(),
            'iter_step': self.iter_step,
        }
        
        jt.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pkl'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_depth_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = jt.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            batch_now = rays_o_batch.shape[0]
            pred_rgb = jt.zeros((batch_now, 3))
            pred_norm = jt.zeros((batch_now, 3))
            pred_dep = jt.zeros((batch_now))


            if render_out != None:
                valid_mask = render_out['valid_mask']
                assert(valid_mask.shape[0] == render_out['comp_rgb'].shape[0])
                pred_rgb[valid_mask] =  render_out['comp_rgb']
                pred_norm[valid_mask] =  render_out['comp_normal']
                pred_dep[valid_mask] =  render_out['depth']
                
            out_rgb_fine.append(pred_rgb.detach().numpy())
            out_normal_fine.append(pred_norm.detach().numpy())
            out_depth_fine.append(pred_dep.detach().numpy())

            # if feasible('comp_rgb'):
            #     out_rgb_fine.append(render_out['comp_rgb'].detach().numpy())
            #     # print("validate: ",render_out['comp_rgb'].shape, render_out['comp_rgb'].sum())

            # if feasible('comp_normal'):
            #     out_normal_fine.append(render_out['comp_normal'].detach().numpy())
            
            # if feasible('depth'):
            #     out_depth_fine.append(render_out['depth'].detach().numpy())


            
            # if feasible('gradients') and feasible('weights') and feasible('z_vals'):
            #     n_samples = self.renderer.n_samples + self.renderer.n_importance
            #     normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
            #     depths  = render_out['z_vals'] * (render_out['weights'][:, :n_samples])

            #     if feasible('inside_sphere'):
            #         normals = normals * render_out['inside_sphere'][..., None]
            #         depths  = depths * render_out['inside_sphere']

            #     normals = normals.sum(dim=1).detach().numpy()
            #     depths  = depths.sum(dim=1).detach().numpy()

            #     out_normal_fine.append(normals)
            #     out_depth_fine.append(depths)

            del render_out


        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)
            # print("img_fine: ",img_fine.shape, img_fine.sum())

        depth_fine = None
        if len(out_depth_fine) > 0:
            depth_fine = np.concatenate(out_depth_fine, axis=0).reshape([H, W])
            depth_fine = cv.applyColorMap(( depth_fine * 255 ).astype(np.uint8),cv.COLORMAP_JET)
            depth_fine = depth_fine.reshape([H,W,3,1])

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
            
            depth_fine = np.concatenate(out_depth_fine, axis=0).reshape([H, W])
            depth_fine = cv.applyColorMap(( depth_fine * 255 ).astype(np.uint8),cv.COLORMAP_JET)
            depth_fine = depth_fine.reshape([H,W,3,1])

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        # os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        # os.makedirs(os.path.join(self.base_exp_dir, 'depths'), exist_ok=True)

        
        for i in range(img_fine.shape[-1]):
            res = None
            if len(out_rgb_fine) > 0:
                res = np.concatenate([img_fine[..., i], self.dataset.image_at(idx, resolution_level=resolution_level)], axis=1)
            if len(out_normal_fine) > 0:
                res = np.concatenate([res, normal_img[..., i]], axis=1)
            if len(out_depth_fine) > 0:
                res = np.concatenate([res, depth_fine[..., i]], axis=1)
            cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                       res)
                

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = jt.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 255).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=128, threshold=0.0):

        bound_min = jt.float32(self.dataset.object_bbox_min)
        bound_max = jt.float32(self.dataset.object_bbox_max)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, f'meshes_{resolution}'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, f'meshes_{resolution}', '{:0>8d}.ply'.format(self.iter_step)))

    def convert_data(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, jt.Var):
            return data.numpy()
        elif isinstance(data, list):
            return [self.convert_data(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        else:
            raise TypeError('Data must be in type numpy.ndarray, jt.Var, list or dict, getting', type(data))
    
    def export(self):
        mesh = self.renderer.export(self.cfg.export)
        self.save_mesh(
            f"it{self.iter_step}-{self.cfg.geometry.isosurface.method}{self.cfg.geometry.isosurface.resolution}.obj",
            **mesh
        )   
        # for the first run, there is no need to continue after getting the bbox 
        if self.cfg.cal_bbox:
            exit(0)

    
    def save_mesh(self, filename, v_pos, t_pos_idx, v_tex=None, t_tex_idx=None, v_rgb=None):
        v_pos, t_pos_idx = self.convert_data(v_pos), self.convert_data(t_pos_idx)
        if v_rgb is not None:
            v_rgb = self.convert_data(v_rgb)

        import trimesh
        mesh = trimesh.Trimesh(
            vertices=v_pos,
            faces=t_pos_idx,
            vertex_colors=v_rgb
        )
        save_path = self.get_save_path(filename)
        mesh.export(save_path)
        
        # simplify mesh if needed
        # import pymeshlab as ml
        # ms = ml.MeshSet()
        # ms.load_new_mesh(save_path)
        # print(f'{ms.current_mesh().face_number()} faces before simplification')
        # ms.meshing_decimation_quadric_edge_collapse(targetfacenum=499000, qualitythr=.1, preserveboundary=True)
        # print(f'{ms.current_mesh().face_number()} faces after simplification')
        # ms.save_current_mesh(self.get_save_path('simp-'+filename))
        
    
    def get_save_path(self, filename):
        save_path = os.path.join(self.base_exp_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
