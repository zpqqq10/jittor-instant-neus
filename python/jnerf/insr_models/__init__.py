models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    model = models[name](config)
    return model

# 这里是调用自定义的模型
from . import geometry, texture, HashGrid #neus, texture
