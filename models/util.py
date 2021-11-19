from __future__ import print_function
from .resnet import resnet12

def create_model(n_cls):
    """create model by name"""
    model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls)

    return model
