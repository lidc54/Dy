import mxnet as mx
from mxnet import nd
from mxboard import SummaryWriter
import math

gamma = 0.0000125  # about 10w to 1
power = 1
c_rate = -0.9
iter_stop = 4500000
nums_power = 1
advanced_iter = 0.5  # defalut is 1:not to advanced
zoom = 5.0
kept_in_kernel = 3
alpha = 1  # harder punishment for loss
HXW = 112 * 96


# compressed_root = 'log_4dy_Ns3/transfed'


class global_sw(object):
    def __init__(self):
        super(global_sw, self).__init__()
        self.sw = None

    def set_sw(self, file):
        self.sw = SummaryWriter(logdir=file, flush_secs=5)
        return self.sw


gls = global_sw()


class mask_param(object):
    def __init__(self, iter=0):
        super(mask_param, self).__init__()
        self.netMask = {}
        self.iter = iter
        self.Is_kept_ratio = 2  # prefer the nums in kernel equal to kept_int_kernel
        self.kept_ratio = 0.0
        self.thr = 20000

    def set_param(self, keys, ctx=mx.cpu()):
        self.netMask = dict(zip(keys, nd.array([1] * len(keys), ctx=ctx)))

    def get_kept_ratio(self):
        if self.iter > self.thr:
            self.kept_ratio = self.Is_kept_ratio * \
                              (1 - math.pow(1 + 10 * gamma * (self.iter - self.thr), -power))
        return self.kept_ratio

    # def get_ratio(self):
    #     # set ratio of importance of this item
    #     if self.iter > self.thr:
    #         ratio = math.pow(1 + 10 * gamma * (self.iter - self.thr), -power)
    #         return ratio * 2
    #     return 0

    def load_param(self, mp, ctx=mx.cpu()):
        self.iter = mp.iter
        for k, v in mp.netMask.items():
            self.netMask[k] = v.as_in_context(ctx)


# mask and iter times for convlution of BatchNorm
# so mask type has a scope of ['conv','bn']
global_param = mask_param()

prefix = ['net.0.a0.alpha', 'net.3.a0.alpha', 'net.1.conv0.weight', 'net.7.conv1.weight',
          'net.1.conv1.bias', 'net.4.conv1.bias', 'net.0.conv0.bias', 'net.7.conv1.bias',
          'net.7.a1.alpha', 'net.4.a2.alpha', 'net.2.conv2.weight', 'net.1.conv0.bias',
          'net.1.conv1.weight', 'net.3.a1.alpha', 'net.4.conv2.weight', 'net.3.conv0.bias',
          'net.3.a2.alpha', 'net.0.a1.alpha', 'net.3.conv2.weight', 'net.0.conv0.weight',
          'net.8.weight', 'net.6.a1.alpha', 'net.3.conv0.weight', 'net.6.conv2.weight',
          'f6.weight', 'net.6.conv1.weight', 'net.5.conv1.weight', 'net.1.a2.alpha',
          'net.2.conv1.weight', 'net.2.conv1.bias', 'net.0.conv2.bias', 'net.7.a0.alpha',
          'net.6.a2.alpha', 'net.1.a0.alpha', 'net.2.conv2.bias', 'net.4.conv1.weight',
          'net.6.conv2.bias', 'net.5.a1.alpha', 'net.2.a1.alpha', 'net.2.a2.alpha',
          'net.1.conv2.bias', 'net.3.conv1.weight', 'net.0.a2.alpha', 'net.0.conv2.weight',
          'net.6.conv1.bias', 'net.1.a1.alpha', 'net.5.conv2.weight', 'net.5.conv2.bias',
          'net.3.conv2.bias', 'net.1.conv2.weight', 'net.5.a2.alpha', 'net.0.conv1.bias',
          'net.5.conv1.bias', 'net.7.conv0.weight', 'net.7.conv0.bias', 'net.0.conv1.weight',
          'net.4.conv2.bias', 'net.7.conv2.bias', 'net.4.a1.alpha', 'net.7.conv2.weight',
          'net.3.conv1.bias', 'net.8.bias', 'net.7.a2.alpha']

collect = ['spherenet200_conv0_weight', 'spherenet200_conv0_bias', 'spherenet200_mprelu0_alpha',
           'spherenet200_conv1_weight', 'spherenet200_conv1_bias', 'spherenet200_mprelu1_alpha',
           'spherenet200_conv2_weight', 'spherenet200_conv2_bias', 'spherenet200_mprelu2_alpha',
           'spherenet200_conv3_weight', 'spherenet200_conv3_bias', 'spherenet200_mprelu3_alpha',
           'spherenet200_conv4_weight', 'spherenet200_conv4_bias', 'spherenet200_mprelu4_alpha',
           'spherenet200_conv5_weight', 'spherenet200_conv5_bias', 'spherenet200_mprelu5_alpha',
           'spherenet200_conv6_weight', 'spherenet200_conv6_bias', 'spherenet200_mprelu6_alpha',
           'spherenet200_conv7_weight', 'spherenet200_conv7_bias', 'spherenet200_mprelu7_alpha',
           'spherenet200_conv8_weight', 'spherenet200_conv8_bias', 'spherenet200_mprelu8_alpha',
           'spherenet200_conv9_weight', 'spherenet200_conv9_bias', 'spherenet200_mprelu9_alpha',
           'spherenet200_conv10_weight', 'spherenet200_conv10_bias', 'spherenet200_mprelu10_alpha',
           'spherenet200_conv11_weight', 'spherenet200_conv11_bias', 'spherenet200_mprelu11_alpha',
           'spherenet200_conv12_weight', 'spherenet200_conv12_bias', 'spherenet200_mprelu12_alpha',
           'spherenet200_conv13_weight', 'spherenet200_conv13_bias', 'spherenet200_mprelu13_alpha',
           'spherenet200_conv14_weight', 'spherenet200_conv14_bias', 'spherenet200_mprelu14_alpha',
           'spherenet200_conv15_weight', 'spherenet200_conv15_bias', 'spherenet200_mprelu15_alpha',
           'spherenet200_conv16_weight', 'spherenet200_conv16_bias', 'spherenet200_mprelu16_alpha',
           'spherenet200_conv17_weight', 'spherenet200_conv17_bias', 'spherenet200_mprelu17_alpha',
           'spherenet200_conv18_weight', 'spherenet200_conv18_bias', 'spherenet200_mprelu18_alpha',
           'spherenet200_conv19_weight', 'spherenet200_conv19_bias', 'spherenet200_mprelu19_alpha',
           'spherenet200_dense0_weight', 'spherenet200_dense0_bias', 'spherenet200_anglelinear0_weight']


def get_params():
    import pickle
    loaded_param = "../model_paramers/global.param"
    with open(loaded_param)as f:
        sv = pickle.load(f)
    print('o')


if __name__ == "__main__":
    get_params()
