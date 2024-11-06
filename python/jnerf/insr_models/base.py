# import torch
# import torch.nn as nn
import jittor as jt
from jittor import Module
from jittor import nn

# from utils.misc import get_rank

class BaseModel(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.rank = get_rank()
        self.setup()
    
    def setup(self):
        raise NotImplementedError
    
    def update_step(self, epoch, global_step):
        pass
    
    def train(self, mode=True):
        return super().train(mode=mode)
    
    def eval(self):
        return super().eval()
    
    def regularizations(self, out):
        return {}
    
    # @torch.no_grad()
    def export(self, export_config):
        return {}
