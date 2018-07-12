import mxnet.gluon.nn as nn
import mxnet.gluon as gluon
from mxnet.gluon import HybridBlock
from mxnet import nd
from mxnet.gluon.model_zoo import vision
import mxnet as mx
import pickle
import math
from sphere_net import SphereNet20


@mx.init.register
class myInitializer(mx.init.Initializer):
    def __init__(self, weight, bias=None):
        super(myInitializer, self).__init__()
        self.weight = weight
        self.bias = bias

    def _init_weight(self, _, arr):
        arr[:] = self.weight

    def _init_bias(self, _, arr):
        arr[:] = self.bias


def myinit(ctx, net, *args):
    with open("/home/ldc/work/caffe_ssd/models/caffemodel.pkl", "rb") as f:
        params = pickle.load(f)
    same_layer = map(str, [0, 1, 3, 7])
    block_index = 0
    res_index = 1

    print '-' * 10
    # only for fc6 asoftloss
    net.f6.collect_params().initialize(ctx=ctx)
    for name, resblock in net.net._children.items():
        print name, ',',
        if name == "8":
            resblock.initialize(init=myInitializer(params['fc5.weight']), ctx=ctx)
            resblock.bias.initialize(init=mx.init.Constant(params['fc5.bias']), force_reinit=True, ctx=ctx)
        elif name == "9":
            resblock.initialize(ctx=ctx)
        else:
            if name in same_layer:
                block_index += 1
                res_index = 1
                resblock.conv0.initialize(init=myInitializer(params['conv%d_%d.weight' % (block_index, res_index)]),
                                          ctx=ctx)
                resblock.conv0.bias.initialize(
                    init=mx.init.Constant(params['conv%d_%d.bias' % (block_index, res_index)]),
                    force_reinit=True, ctx=ctx)
                print 'conv%d_%d.weight' % (block_index, res_index)
                resblock.a0.initialize(
                    init=mx.init.Constant(params['relu%d_%d' % (block_index, res_index)][0].reshape([1, -1, 1, 1])),
                    force_reinit=True, ctx=ctx)

                res_index += 1
            resblock.conv1.initialize(init=myInitializer(params['conv%d_%d.weight' % (block_index, res_index)],
                                                         params['conv%d_%d.bias' % (block_index, res_index)]),
                                      ctx=ctx)
            resblock.a1.initialize(
                init=mx.init.Constant(params['relu%d_%d' % (block_index, res_index)][0].reshape([1, -1, 1, 1])),
                force_reinit=True, ctx=ctx)
            print 'conv%d_%d.weight' % (block_index, res_index)
            res_index += 1
            resblock.conv2.initialize(init=myInitializer(params['conv%d_%d.weight' % (block_index, res_index)],
                                                         params['conv%d_%d.bias' % (block_index, res_index)]),
                                      ctx=ctx)
            resblock.a2.initialize(
                init=mx.init.Constant(params['relu%d_%d' % (block_index, res_index)][0].reshape([1, -1, 1, 1])),
                force_reinit=True, ctx=ctx)
            print 'conv%d_%d.weight' % (block_index, res_index)

            res_index += 1
    # print net


if __name__ == "__main__":
    net = SphereNet20()
    myinit(mx.cpu(), net)
    net.save_params("model_paramers/spherenet_ft_model")

    print('o')
