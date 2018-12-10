"""
dynamic add extra weight for classifier

and loss equals to two levels classifition
"""
import mxnet.gluon.nn as nn
import numpy as np

from layers.sphere_net import AngleLinear
class layer_expand(nn.Block):
    def __init__(self,num_classes,input=512):
        super(layer_expand,self).__init__()
        self.fc= AngleLinear(input, num_classes)
    def forward(self, *args):
        pass

class trans_load(object):
    def __init__(self,net,class_idx,members_list):
        self.fc=net
    def  read(self):
        params = self.fc.collect_params()
        value=params.values()#[-1]
        ctx=value.context()
        weight=value[:,class_idx].asnumpy()
        weight=np.tile(weight,(1,members))
        #build a new classifier
        classifier=layer_expand(members)
        