import jittor as jt
import numpy as np
import math

def enlarge(x: jt.Var, size: int):
    if x.shape[0] < size:
        y = jt.empty([size],x.dtype)
        x.assign(y)

class BoundingBox():
    def __init__(self,min=[0,0,0],max=[0,0,0]) -> None:
        self.min=np.array(min)
        self.max=np.array(max)
    def inflate(self,amount:float):
        self.min-=amount
        self.max+=amount
        pass

def round_floats(number, digits = 1):
    times = 10 ** digits
    if number > 0:
        return math.ceil(number * times) / times
    elif number < 0:
        return math.floor(number * times) / times
    else:
        # directly return 0
        return 0 