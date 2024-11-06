import jittor as jt
from jittor import Module
from jittor import nn
from jnerf.utils.config import get_cfg
from jittor import Function
import numpy as np
import random
import numpy as np
from tqdm import tqdm
import time
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
import os
from jnerf import insr_models as insr_models


current_path = os.getcwd()
HEADER = os.path.join(current_path, 'cuda_header')
print("occupancy HEADER path: ",HEADER)


class Occupancy_Grid(Module):
    def __init__(self, bbox_corner, bbox_size, _resolution):
        super().__init__()

        self.bbox_corner = bbox_corner
        self.bbox_size = bbox_size
        self.resolution = _resolution
        print("occupancy grid config : ", _resolution, 'bbox corner / size : ',self.bbox_corner, self.bbox_size)

        occupancy_grid = jt.zeros((self.resolution, self.resolution, self.resolution))
        occupancy_grid = (occupancy_grid == 0)

        # all true in inital
        self.occupancy_grid = occupancy_grid
        self.grid_counts = jt.zeros((self.resolution, self.resolution, self.resolution)).int()
        self.grid_weight = jt.zeros((self.resolution, self.resolution, self.resolution)).int()
        

    def counts_init(self):
        self.grid_counts = jt.zeros((self.resolution, self.resolution, self.resolution)).int() 
        self.grid_weight = jt.zeros((self.resolution, self.resolution, self.resolution)).int()

    # to determine if the point is inside the bounding box
    def points_mask(self, pts):
        bbox_corner = self.bbox_corner

        bbox_corner = jt.Var(self.bbox_corner)
        bbox_size = jt.Var(self.bbox_size)
        batch_size = pts.shape[0]

        occupied_gird = self.occupancy_grid
        rx = occupied_gird.shape[0]
        ry = occupied_gird.shape[1]
        rz = occupied_gird.shape[2]

        header_path = HEADER
        proj_options[f"FLAGS: -I{header_path}"]=1

        out = jt.ones(pts.shape[0]).int()
        out_mask = jt.ones(pts.shape[0]).int()

        out = jt.code(out.shape, out.dtype, [pts, occupied_gird, bbox_corner, bbox_size, out_mask],
        cuda_header='#include "helper.h"',
        cuda_src=f"""
        #define MIN_VALUE(a,b) a < b ? a : b
        #define NUM_THREAD 512
        #define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)

        @alias(pts, in0)
        @alias(occupied_gird, in1)
        @alias(block_corner, in2)
        @alias(block_size, in3)
        @alias(out_mask, in4)

        @alias(out, out0)

        int batch_size = {batch_size};
        int rx = {rx};
        int ry = {ry};
        int rz = {rz};


        points_in_grid_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
            (float3*)pts_p,
            (bool*)occupied_gird_p,
            (float3*)block_corner_p,
            (float3*)block_size_p,
            (int*)out_mask_p,
            rx, ry, rz,
            batch_size);
        """)       
        
        out.compile_options=proj_options
        out.sync()

        if out_mask.sum().isnan() == True:
            print("???? out_mask: ",out_mask, out_mask.shape, out_mask.sum())
            exit(0)

        return out_mask


    # rays are all within the masks
    # to determine the interested points in the grid according to the masks roughly
    def grid_pruning(self, rays_o, rays_d):
        counts = self.grid_counts

        bbox_corner = jt.Var(self.bbox_corner)
        bbox_size = jt.Var(self.bbox_size)

        rx = self.grid_counts.shape[0]
        ry = self.grid_counts.shape[1]
        rz = self.grid_counts.shape[2]
        batch_size = rays_o.shape[0]
        
        assert(counts.shape[0] == rx)
        assert(counts.shape[1] == ry)
        assert(counts.shape[2] == rz)

        out = jt.zeros((rx, ry, rz)).int()
        header_path = HEADER
        proj_options[f"FLAGS: -I{header_path}"]=1
        
        out = jt.code(out.shape, out.dtype, [rays_o, rays_d, bbox_corner, bbox_size, counts],
        cuda_header='#include "helper.h"',
        cuda_src=f"""
        #define MIN_VALUE(a,b) a < b ? a : b
        #define NUM_THREAD 512
        #define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)

        @alias(rays_o, in0)
        @alias(rays_d, in1)
        @alias(block_corner, in2)
        @alias(block_size, in3)
        @alias(counts, in4)

        @alias(out, out0)

        int batch_size = {batch_size};
        int rx = {rx};
        int ry = {ry};
        int rz = {rz};

        /**/

        grid_pruning_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
            (float3*)rays_o_p,
            (float3*)rays_d_p,
            (float3*)block_corner_p,
            (float3*)block_size_p,
            (int*)counts_p,
            rx, ry, rz,
            batch_size);
        
        """)            

        
        out.compile_options=proj_options
        out.sync()

        if counts.sum().isnan() == True:
            print("???? counts: ",counts, counts.shape, counts.sum())
            exit(0)

        # return counts
        self.grid_counts = counts
        
    def save(self, grid_path, weight_path = None):

        print('save occupancy_grid at: ',grid_path, weight_path)
        np.save(grid_path, self.occupancy_grid.detach().numpy())
        if weight_path is not None:
            np.save(weight_path, self.grid_weight.detach().numpy())
        
    def load(self, grid_path, weight_path = None):

        print('load exsiting occupancy_grid at: ',grid_path, weight_path)
        x = np.load(grid_path)
        self.occupancy_grid = jt.Var(x)
        if weight_path is not None:
            x = np.load(weight_path)
            self.grid_weight = jt.Var(x)


    def occupied_grid_init(self):
        self.occupancy_grid =  (self.grid_counts < 50)
        print('occupied_gird_init : ', self.occupancy_grid.sum(), self.occupancy_grid.shape)

        self.grid_counts = None

    def sample_z_vals(self, rays_o, rays_d, n_samples):
        # sample_points_kernel

        batch_size = rays_o.shape[0]

        bbox_corner = jt.Var(self.bbox_corner)
        bbox_size = jt.Var(self.bbox_size)

        occupied_gird = self.occupancy_grid

        rx = occupied_gird.shape[0]
        ry = occupied_gird.shape[1]
        rz = occupied_gird.shape[2]

        num_sample = n_samples
        z_vals = jt.zeros((batch_size, num_sample))
        out = jt.zeros((rx, ry, rz)).int()

        header_path = HEADER
        proj_options[f"FLAGS: -I{header_path}"]=1
        
        
        out = jt.code(out.shape, out.dtype, 
        [rays_o, rays_d, z_vals, bbox_corner, bbox_size, occupied_gird],
        cuda_header='#include "helper.h"',
        cuda_src=f"""
        #define MIN_VALUE(a,b) a < b ? a : b
        #define NUM_THREAD 512
        #define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)

        @alias(rays_o, in0)
        @alias(rays_d, in1)
        @alias(z_vals, in2)
        @alias(block_corner, in3)
        @alias(block_size, in4)
        @alias(occupied_gird, in5)

        @alias(out, out0)

        int batch_size = {batch_size};
        int rx = {rx};
        int ry = {ry};
        int rz = {rz};
        int num_sample = {num_sample}; 


        sample_points_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
            (float3*)rays_o_p,
            (float3*)rays_d_p,
            num_sample, (float*)z_vals_p,
            (float3*)block_corner_p,
            (float3*)block_size_p,
            (bool*)occupied_gird_p,
            rx, ry, rz,
            batch_size);
        """)       
        
        out.compile_options=proj_options
        out.sync()

        if z_vals.sum().isnan() == True:
            print("???? z_vals: ",z_vals, z_vals.shape, z_vals.sum())
            exit(0)
        
        return z_vals

    # accumulate the weights of the points in the grid
    # to prune useless vertices
    def update_grid_weight(self, iter, pts, weights):
        batch_size = pts.shape[0]

        bbox_corner = jt.Var(self.bbox_corner)
        bbox_size = jt.Var(self.bbox_size)

        occupied_gird = self.occupancy_grid
        rx = occupied_gird.shape[0]
        ry = occupied_gird.shape[1]
        rz = occupied_gird.shape[2]

        header_path = HEADER
        proj_options[f"FLAGS: -I{header_path}"]=1

        out = jt.zeros(pts.shape)

        grid_w = self.grid_weight
        
        out = jt.code(out.shape, out.dtype, [pts, weights, grid_w, bbox_corner, bbox_size],
        cuda_header='#include "helper.h"',
        cuda_src=f"""
        #define MIN_VALUE(a,b) a < b ? a : b
        #define NUM_THREAD 512
        #define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)

        @alias(pts, in0)
        @alias(weights, in1)
        @alias(grid_w, in2)
        @alias(block_corner, in3)
        @alias(block_size, in4)

        @alias(out, out0)

        int batch_size = {batch_size};
        int rx = {rx};
        int ry = {ry};
        int rz = {rz};


        update_grid_weight_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
            (float3*)pts_p,
            (float*)weights_p,
            (int*)grid_w_p,
            (float3*)block_corner_p,
            (float3*)block_size_p,
            rx, ry, rz,
            batch_size);
        """)       
        
        out.compile_options=proj_options
        out.sync()

        if grid_w.sum().isnan() == True:
            print("???? z_vals: ",grid_w, grid_w.shape, grid_w.sum())
            exit(0)

        self.grid_weight = grid_w


    def update_occupancy(self, iter):
        drop_mask = (self.grid_weight.view(-1) == 0).nonzero()[:, 0]

        rx, ry, rz = self.occupancy_grid.shape
        occupancy_grid = self.occupancy_grid.view(-1)
        occupancy_grid[drop_mask] = 0

        self.occupancy_grid = occupancy_grid.reshape(rx, ry, rz)
        print('update occupancy_grid at iter ',iter, ' sum : ',self.occupancy_grid.sum())

        ### reset grid weight.
        self.grid_weight = jt.zeros((self.resolution, self.resolution, self.resolution)).int()

    def clean_weight(self):
        self.grid_weight = jt.zeros((self.resolution, self.resolution, self.resolution)).int()
        

    def vis_grid(self, path):
        def save_obj(pts, path):

            obj_path = path
            obj_file = open(obj_path, 'w')
            N = pts.shape[0]
            for i in tqdm(range(N)):
                # c = jt.Var([1., 0., 0.])
                c = jt.rand((3))
                c = c / 2 + 0.5
                obj_file.write("v {0} {1} {2} {3} {4} {5}\n".format(pts[i,0],pts[i,1],pts[i,2], c[0], c[1], c[2]))
            obj_file.close()

        out_ply_path = path
        print('save grid at : ',out_ply_path)

        loc = (self.grid_counts == 0 ).nonzero()

        # pts = ((loc + 0.5) / self.resolution * 2.) + -1.
        pts = ((loc + 0.5) / self.resolution * self.bbox_size) + self.bbox_corner
        save_obj(pts, path)

        