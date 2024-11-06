import jittor
import jittor as jt
from jittor import Module
from jittor import nn

from collections import defaultdict
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS

import math

'''

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply
'''

def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: jt.where(x > 0.0031308, jt.pow(jt.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name == 'trunc_exp':
        raise NotImplementedError
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name == 'sigmoid':
        return lambda x: jt.sigmoid(x)
    elif name == 'tanh':
        return lambda x: jt.tanh(x)
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'softplus':
        return nn.Softplus(beta=100)
    else:
        raise NotImplementedError
        
def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat

class CompositeEncoding(Module):
    def __init__(self, encoding, include_xyz=False, xyz_scale=1., xyz_offset=0.):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        self.n_output_dims = int(self.include_xyz) * self.encoding.input_dims + self.encoding.out_dim
        # print("CompositeEncoding :",include_xyz, self.n_output_dims)
    def execute(self, x, *args):
        # print("? :",x, self.xyz_scale, self.xyz_offset)
        return self.encoding(x, *args) if not self.include_xyz else jt.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1)

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

# handle if include xyz
# 这里是调用jnerf的encoder
def get_encoding(n_input_dims, config):
    if config.pos_encoder != None:
        print(f'pos_encoder {config.pos_encoder.type} create')
        encoding = build_from_cfg(config.pos_encoder, ENCODERS)
    elif config.dir_encoding_config != None:
        print(f'dir encoder {config.dir_encoding_config.type} create')
        encoding = build_from_cfg(config.dir_encoding_config, ENCODERS)
    
    
    # print(" encoding check : ",encoding.input_dim, encoding.out_dim)

    encoding = CompositeEncoding(encoding, include_xyz=config.get('include_xyz', False), xyz_scale=1., xyz_offset=0.)
    
    return encoding



class VanillaMLP(Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)

        self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False), self.make_activation()]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config['output_activation'])
    
    def execute(self, x):
        y = self.layers(x.float())
        z = self.output_activation(y)
        return z
    
    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True) # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                nn.init.constant_(layer.bias, -self.sphere_init_radius)
                nn.init.gauss_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                nn.init.constant_(layer.bias, 0.0)
                nn.init.constant_(layer.weight[:, 3:], 0.0)
                nn.init.gauss_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                nn.init.constant_(layer.bias, 0.0)
                nn.init.gauss_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if self.weight_norm:
            raise NotImplementedError
        return layer   

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU()

def get_mlp(n_input_dims, n_output_dims, config):
    if config.otype == 'VanillaMLP':
        network = VanillaMLP(n_input_dims, n_output_dims, config)
    else:
        print("Error : mlp type not VanillaMLP.")
        raise NotImplementedError
    return network


def update_module_step(m, epoch, global_step):
    if hasattr(m, 'update_step'):
        m.update_step(epoch, global_step)


def chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, jt.Var):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(*[arg[i:i+chunk_size] if isinstance(arg, jt.Var) else arg for arg in args], **kwargs)
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, jt.Var):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(f'Return value of func must be in type [jt.Var, list, tuple, dict], get {type(out_chunk)}.')
            exit(1)
        for k, v in out_chunk.items():
            # v = v if torch.is_grad_enabled() else v.detach()
            v = v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)
    
    if out_type is None:
        return

    out = {k: jt.cat(v, dim=0) for k, v in out.items()}
    if out_type is jt.Var:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


### contract from pts in bbox to (0, 1)
def contract_to_unisphere(x, bbox_corner, bbox_size):
    
    # x = scale_anything(x, (-radius, radius), (0, 1))
    radius = bbox_size.max() * 0.5
    bbox_center = bbox_corner + 0.5 * bbox_size

    x = jt.stack([
        scale_anything(x[..., 0], (bbox_center[0] - radius, bbox_center[0] + radius), (0, 1)),
        scale_anything(x[..., 1], (bbox_center[1] - radius, bbox_center[1] + radius), (0, 1)),
        scale_anything(x[..., 2], (bbox_center[2] - radius, bbox_center[2] + radius), (0, 1)),
    ], dim = -1)
    
    return x