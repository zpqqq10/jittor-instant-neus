import jittor as jt
from jittor import nn
from jnerf.utils.config import get_cfg
from jittor import Function
import numpy as np
import random
import numpy as np

from jnerf.ops.code_ops.global_vars import global_headers,proj_options
import os
from jnerf import insr_models as insr_models


current_path = os.getcwd()
HEADER = os.path.join(current_path, 'cuda_header')
print("ray_aabb HEADER path: ",HEADER)

class Ray_AABB(Function):
    def execute(self, rays_o, rays_d, bbox_center, bbox_size):
        
        batch_size = rays_o.shape[0]
        bounds = jt.full((batch_size, 2), 0, dtype=jt.float32) #, device=points.device
        
        aabb_center = bbox_center
        aabb_size = bbox_size

        header_path = HEADER
        proj_options[f"FLAGS: -I{header_path}"]=1

        bounds = jt.code(bounds.shape, bounds.dtype, [rays_o, rays_d, aabb_center, aabb_size],
        cuda_header='#include "helper.h"',
        cuda_src=f"""
        #define MIN_VALUE(a,b) a < b ? a : b
        #define NUM_THREAD 512
        #define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)

        @alias(rays_o, in0)
        @alias(rays_d, in1)
        @alias(aabb_center, in2)
        @alias(aabb_size, in3)

        @alias(bounds, out0)

        int batch_size = {batch_size};

        ray_aabb_intersection_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
            (float3*)rays_o_p,
            (float3*)rays_d_p,
            (float3*)aabb_center_p,
            (float3*)aabb_size_p,
            (float2*)bounds_p,
            batch_size
        );
        """)            
        bounds.compile_options=proj_options

        bounds.sync()

        if bounds.sum().isnan() == True:
            print("???? bounds: ",bounds, bounds.shape, bounds.sum())
            exit(0)

        return bounds
