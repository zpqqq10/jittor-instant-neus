import jittor as jt
from jittor import nn
from jnerf.utils.config import get_cfg
from jittor import Function
import numpy as np
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
import os
from jnerf import insr_models as insr_models
from jnerf.insr_models.base import BaseModel

current_path = os.getcwd()
HEADER = os.path.join(current_path, 'cuda_header')
print("HEADER path: ",HEADER)

class HashEncoding(Function):
    def execute(self, points, features, resolution):
        assert(resolution.shape[1] == 3)
        
        batch_size = points.shape[0]
        n_levels = features.shape[0]
        hashmap_size = features.shape[1]

        outputs = jt.full((batch_size, n_levels, 2), 0, dtype=jt.float32) #, device=points.device

        # ctx.save_for_backward(points, features, resolution)
        self.save_vars = points, features, resolution

        header_path = HEADER
        proj_options[f"FLAGS: -I{header_path}"]=1

        
        outputs = jt.code(outputs.shape, points.dtype, [points, features, resolution],
        cuda_header='#include "hashgrid.h"',
        cuda_src=f"""
        #define MIN_VALUE(a,b) a < b ? a : b
        #define NUM_THREAD 512
        #define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)
        @alias(points, in0)
        @alias(features, in1)
        @alias(resolution, in2)
        @alias(outputs, out0)

        int batch_size = {batch_size};
        int n_levels = {n_levels};
        int hashmap_size = {hashmap_size};

        embedding_forward_kernel<<<dim3(NUM_BLOCK(batch_size), n_levels), NUM_THREAD>>>(
            (float3*)points_p,
            (float2*)outputs_p,
            (float2*)features_p,
            n_levels, hashmap_size, 
            (int3*)resolution_p,
            batch_size
        );
        """)            
        outputs.compile_options=proj_options

        outputs.sync()

        if outputs.sum().isnan() == True:
            print("???? outputs: ",outputs, outputs.shape, outputs.sum())
            exit(0)


        return outputs

    def grad(self, grad_in):
        points, features, resolution = self.save_vars
        return HashEncodingBackward.apply(grad_in, points, features, resolution)

def HashEmbedding(points, features, resolution):
    return HashEncoding.apply(points, features, resolution)

# for calculating double backward (eikonal loss requires second order gradient)
class HashEncodingBackward(Function):
    
    def execute(self, grad_in, points, features, resolution):
        self.save_var = grad_in, points, features, resolution

        batch_size = points.shape[0]
        n_levels = features.shape[0]
        hashmap_size = features.shape[1]

        grad_points = jt.zeros_like(points, dtype=jt.float32) #, device=points.device
        grad_features = jt.zeros_like(features, dtype=jt.float32) #, device=points.device

        header_path = HEADER
        proj_options[f"FLAGS: -I{header_path}"]=1
        
        output =  jt.code( grad_points.shape, points.dtype, 
                        [points, grad_in, features, resolution, grad_points, grad_features],
        cuda_header='#include "hashgrid.h"',
        cuda_src=f"""
        #define MIN_VALUE(a,b) a < b ? a : b
        #define NUM_THREAD 512
        #define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)
        
        @alias(points, in0)
        @alias(grad_in, in1)
        @alias(features, in2)
        @alias(resolution, in3)
        @alias(grad_points, in4)
        @alias(grad_features, in5)

        @alias(output, out0)

        int batch_size = {batch_size};
        int n_levels = {n_levels};
        int hashmap_size = {hashmap_size};

        embedding_backward_kernel<<<dim3(NUM_BLOCK(batch_size), n_levels), NUM_THREAD>>>(
            (float3*)points_p,
            (float2*)grad_in_p,
            (float3*)grad_points_p,
            (float2*)grad_features_p,
            (float2*)features_p,
            n_levels, hashmap_size,
            (int3*)resolution_p,
            batch_size
        );
        """)            

        output.compile_options=proj_options
        output.sync()
        
        if grad_points.sum().isnan() == True or grad_features.sum().isnan() == True:
            print("???? grad_points : ",grad_points, grad_points.shape, grad_points.sum())
            print("???? grad_features: ",grad_features, grad_features.shape, grad_features.sum())

        return grad_points, grad_features, None


    def grad(self, d_grad_points, d_grad_features, d_grad_temp):
        grad_in, points, features, resolution = self.save_var

        batch_size = points.shape[0]
        n_levels = features.shape[0]
        hashmap_size = features.shape[1]


        d_grad_in = jt.zeros_like(grad_in, dtype=jt.float32) # device=points.device

        header_path = HEADER
        proj_options[f"FLAGS: -I{header_path}"]=1
        
        d_grad_in = jt.code(d_grad_in.shape, points.dtype, [points, d_grad_points, features, resolution],
        cuda_header='#include "hashgrid.h"',
        cuda_src=f"""
        #define MIN_VALUE(a,b) a < b ? a : b
        #define NUM_THREAD 512
        #define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)
        
        @alias(points, in0)
        @alias(d_grad_points, in1)
        @alias(features, in2)
        @alias(resolution, in3)

        @alias(d_grad_in, out0)

        int batch_size = {batch_size};
        int n_levels = {n_levels};
        int hashmap_size = {hashmap_size};

        embedding_backward_backward_kernel<<<dim3(NUM_BLOCK(batch_size), n_levels), NUM_THREAD>>>(
            (float3*)points_p,
            (float3*)d_grad_points_p,
            (float2*)d_grad_in_p,
            (float2*)features_p,
            n_levels, hashmap_size, 
            (int3*)resolution_p,
            batch_size
        );
        """)            
        d_grad_in.compile_options=proj_options
        d_grad_in.sync()


        if d_grad_in.sum().isnan() == True:
            print("???? d_grad_in : ",d_grad_in, d_grad_in.shape, d_grad_in.sum())
            exit(0)

        return d_grad_in, None, None, None

    
@insr_models.register('HashGrid')
class HashGrid(BaseModel):
    def setup(self, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=32, finest_resolution=2048,
                 init_mode="kaiming", n_pos_dims = 3):
        assert self.config.n_features_per_level == 2, "we only support dim=2"
        self.xyz_scale = 2.
        self.xyz_offset = -1.
        self.n_levels = self.config.n_levels
        self.n_features_per_level = self.config.n_features_per_level
        self.log2_hashmap_size = self.config.log2_hashmap_size
        self.base_resolution = self.config.base_resolution
        self.finest_resolution = self.config.finest_resolution
        self.out_dim = self.n_levels * self.n_features_per_level
        self.include_xyz = self.config.include_xyz

        # that is, per_level_scale, calculated before initialization
        self.b = jt.exp((jt.log(self.finest_resolution)-jt.log(self.base_resolution))/(self.n_levels-1))

        self.resolution = []
        for i in range(self.n_levels):
            self.resolution += [(self.base_resolution * self.b**i).int()]
        self.resolution = jt.stack(self.resolution, 0).expand(16,3).contiguous()
        self.resolution.requires_grad = False
                                    
        self.features = jt.zeros(self.n_levels, 2**self.log2_hashmap_size, self.n_features_per_level,
                                    dtype=jt.float32)
        self.features.requires_grad = True

        if init_mode == 'kaiming':
            nn.init.kaiming_normal_(self.features)
        elif init_mode == 'xavier':
            # nn.init.xavier_normal_(self.features)
            nn.init.xavier_uniform_(self.features)
        elif init_mode == 'uniform':
            nn.init.uniform_(self.features, -1e-4, 1e-4)
        print(f"{init_mode} init feature")

        # control the level mask, and updated in grometry
        self.current_level = n_levels
        self.input_dim = 3
        self.n_output_dims = self.n_features_per_level * self.n_levels
        if self.include_xyz:
            self.n_output_dims += 3

    def execute(self, x):
        """
        x ... x 3
        return ... x 32 
        """
        ori_shape = x.shape[:-1]
        features = HashEmbedding(x.reshape(-1,3), self.features, self.resolution)
        features = features.reshape(*list(ori_shape), self.n_levels*self.n_features_per_level)
        
        level_mask = jt.zeros(self.n_levels * self.n_features_per_level)
        level_mask[: self.current_level * self.n_features_per_level] = 1.
        features = features * level_mask
        
        if self.include_xyz:
            features = jt.cat([x * self.xyz_scale + self.xyz_offset, features], dim=-1)

        return features

# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # setup_seed(42)
    
    cfg=None
    phg = HashGrid(cfg).cuda()
    
    # x = jt.full((1, 3), 0.5, dtype=jt.float32)
    x = jt.rand((65536, 3), dtype=jt.float32)
    
    x.requires_grad = True
    y = phg(x)# 100 x 32
    print('x: ',x, x.shape)
    print('y: ',y, y.shape)

    grad = jt.grad(y, x)
    print('grad:', grad, grad.shape)
