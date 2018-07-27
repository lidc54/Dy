'''
in this code,we will:
1. general convolution can be carried out;
2. special convolution also be carried out.
'''

import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
import numpy as np
from layers.params import HXW
from layers.sphere_net import mPReLU, AngleLinear


class special_conv(nn.Conv2D):
    """
    as a matter of fact, we can only deal with general ones;
    such as stride=1 & kernel=3
    """

    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), in_channels=0, **kwargs):
        numeric_types = (float, int, long)
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,) * 2
        if isinstance(strides, numeric_types):
            strides = (strides,) * 2
        if isinstance(padding, tuple):
            padding = padding[0]
        kernel_size = list(kernel_size)
        kernel_size.sort()
        self.kernel_size = tuple(kernel_size)
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.has_deal = False
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        self.kernel_hw = self.kernels()
        super(special_conv, self).__init__(channels, kernel_size, strides=strides,
                                           padding=padding, dilation=dilation,
                                           in_channels=in_channels, **kwargs)

    def hybrid_forward(self, F, x, weight, bias=None):
        new_weight = weight.reshape(weight.shape[0], -1, 1, 1, 1)
        out = nd.sum(new_weight * self.special_im2col(x)[self.special_offset()], axis=1)
        if not bias is None:
            new_bias = bias.reshape(-1, 1, 1, 1)
            out = out + new_bias
        out = out.transpose((3, 0, 1, 2))
        return out

    def offset(self, kernel_size):
        kernel_h, kernel_w = kernel_size
        dilation_h, dilation_w = self.dilation
        offset_h = (nd.arange(kernel_h)) * dilation_h  # - row
        offset_h = nd.tile(offset_h, (kernel_w, 1)).T.reshape(-1)
        offset_w = (nd.arange(kernel_w)) * dilation_w  # - col
        offset_w = nd.tile(offset_w, (kernel_h, 1)).reshape(-1)
        return offset_h, offset_w

    def special_offset(self):
        name = self.name + '_weight'
        mask = masks[name]
        if not self.has_deal:
            self.has_deal = True
            output, input, k = mask.shape
            kernelsz = self.kernel_size[1] ** 2
            new_idx = np.tile(np.arange(input) * kernelsz, (output, 1)).reshape(output, input, 1)
            mask += new_idx
        return mask.reshape(mask.shape[0], -1)

    def kernels(self):
        kernel = self.kernel_size[1]
        return self.offset((kernel, kernel))

    def special_im2col(self, temp_img):  # , idx_out
        N, C, height, width = temp_img.shape
        offset_h, offset_w = self.kernel_hw
        shape_oh, shape_ow = offset_h.shape, offset_w.shape
        offset_h = offset_h.broadcast_to((C,) + shape_oh).asnumpy().astype('int')
        offset_w = offset_w.broadcast_to((C,) + shape_ow).asnumpy().astype('int')
        shedule = np.tile(np.arange(self.kernel_size[1] ** 2), (C, 1))

        assert isinstance(self.padding, int), 'padding should be a number'
        pad = (0,) * 4 + (self.padding,) * 4
        stride_h, stride_w = map(int, self.strides)
        height -= height % stride_h
        width -= width % stride_w
        data = nd.pad(temp_img, mode="constant", pad_width=pad)
        data = data.transpose((1, 2, 3, 0))

        array_channel = []
        for n in range(C):
            array_kernel = []
            for i in shedule[n]:
                start_h, start_w = offset_h[n, i], offset_w[n, i]
                end_h, end_w = start_h + height, start_w + width
                array_kernel.append(data[n, start_h:end_h:stride_h, start_w:end_w:stride_w, :])
            array_channel.append(nd.stack(*array_kernel))

        sz = array_channel[0].shape
        pit = nd.stack(*array_channel).reshape((-1,) + sz[1:])

        return pit


class base_net(nn.Block):
    def __init__(self, channels, in_channels, kernel_size=(3, 3), fun=special_conv, same_shape=True):
        super(base_net, self).__init__()
        self.same_shape = same_shape
        if not same_shape:
            self.conv0 = fun(channels, kernel_size=kernel_size,
                             in_channels=in_channels, padding=1, strides=2)
            self.a0 = mPReLU(channels)  # nn.PReLU()
        self.conv1 = fun(channels, kernel_size=kernel_size, in_channels=channels, padding=1)
        self.a1 = mPReLU(channels)  # nn.PReLU()
        self.conv2 = fun(channels, kernel_size=kernel_size, in_channels=channels, padding=1)
        self.a2 = mPReLU(channels)  # v nn.PReLU()

    def forward(self, x):
        if not self.same_shape:
            x = self.a0(self.conv0(x))
        out = self.a1(self.conv1(x))
        out = self.a2(self.conv2(out))
        return out + x


class SphereNet20_3(nn.Block):
    # http://ethereon.github.io/netscope/#/gist/20f6ddf70a35dec5019a539a502bccc5
    def __init__(self, num_classes=10574, verbose=False, kernel_size=(1, 3), fun=special_conv, use_bn=True, **kwargs):
        super(SphereNet20_3, self).__init__(**kwargs)
        self.verbose = verbose
        self.feature = True  # weather only fc1 be returned or train with a classifier
        self.has_mask = False  # the global mask for gengeral convolution
        # add name_scope on the outermost Sequential
        in_put = 3
        with self.name_scope():
            # block 1
            self.net = nn.Sequential()
            b1_1 = base_net(64, in_put, fun=fun, kernel_size=kernel_size, same_shape=False)
            in_put = 64

            # block 2
            b2_1 = base_net(128, in_put, fun=fun, kernel_size=kernel_size, same_shape=False)
            in_put = 128
            b2_2 = base_net(128, in_put, fun=fun, kernel_size=kernel_size)

            # block3
            b3_1 = base_net(256, in_put, fun=fun, kernel_size=kernel_size, same_shape=False)
            in_put = 256
            b3_2 = base_net(256, in_put, fun=fun, kernel_size=kernel_size)
            b3_3 = base_net(256, in_put, fun=fun, kernel_size=kernel_size)
            b3_4 = base_net(256, in_put, fun=fun, kernel_size=kernel_size)

            # block 4
            b4_1 = base_net(512, in_put, fun=fun, kernel_size=kernel_size, same_shape=False)
            in_put = 512
            blocks = 4

            in_put = in_put * HXW / (blocks ** 2 * blocks ** 2)
            f5_1 = nn.Dense(512, in_units=in_put)
            self.f6 = AngleLinear(512, num_classes)
            self.net.add(b1_1, b2_1, b2_2, b3_1, b3_2, b3_3, b3_4, b4_1, f5_1)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            # print 'layer', i,
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        if not self.feature:
            out = self.f6(out)
        return out

    def build_mask(self, mask_path):
        self.has_mask = True
        import pickle, os
        # assert ctx is not None, 'ctx is none'
        assert os.path.exists(mask_path), 'the mask doesnot exists'
        global masks
        with open(mask_path)as f:
            masks = pickle.load(f)
        # for k, v in masks.items():
        #     v = v.as_in_context(ctx)


############################################################
# for test
############################################################
def test(ctx=mx.cpu()):
    from mxboard import SummaryWriter
    sw = SummaryWriter(logdir='sphere_dynamic', flush_secs=5)

    net = nn.Sequential()
    b1 = base_net(48, 3, fun=special_conv, kernel_size=(3, 3), same_shape=False)
    b2 = base_net(1, 48, fun=special_conv, kernel_size=(3, 3), same_shape=False)
    fc = nn.Dense(3, in_units=9)
    net.add(b1, b2, fc)
    init_s(net, ctx)

    from mxnet import gluon, autograd
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    for i in range(10000):
        with autograd.record():
            out = net(img)
            loss = nd.sum(nd.abs(out - target))
        loss.backward()
        trainer.step(2)
        sw.add_scalar(tag='loss', value=loss.asscalar(), global_step=i)
        if i % 100 == 0:
            print i, loss.asscalar()
    sw.close()


def init_s(mnet, ctx=mx.cpu()):
    for k, v in mnet.collect_params().items():
        if 'bias' in k or 'alpha' in k:
            v.initialize(mx.initializer.Constant(0.0), ctx=ctx)
        else:
            v.initialize(mx.initializer.Xavier(magnitude=1), ctx=ctx)


def load_s(net, target_net, ctx=mx.cpu()):
    n_params = net._collect_params_with_prefix()
    t_params = target_net._collect_params_with_prefix()
    n_k = n_params.keys()
    t_k = t_params.keys()
    # n_k=net.collect_params().keys()
    # t_k=target_net.collect_params().keys()
    for origin_name, target_name in zip(n_k, t_k):
        # origin_value = net.collect_params()[origin_name]
        # target_value = target_net.collect_params()[target_name]
        # origin_value.set_data(target_value.data().copy())
        n_params[origin_name]._load_init(t_params[target_name]._reduce(), ctx)
        print ''


def test2():
    cov = special_conv(1, kernel_size=(3, 1),
                       in_channels=3, padding=1, strides=2)
    Oconv = nn.Conv2D(1, kernel_size=(3, 1),
                      in_channels=3, padding=1, strides=2)
    init_s(cov)
    load_s(Oconv, cov)
    Oout = Oconv(img)
    out = cov(img)
    print 'weight:', nd.sum(Oconv.weight.data() - cov.weight.data()),
    print 'result:', nd.sum(Oout - out),
    print ''


if __name__ == "__main__":
    ctx = mx.gpu()
    global mask, img, target
    mask = np.zeros((3, 9))
    # mask[range(3), np.random.randint(0, 9, (3, 3))] = 1
    mask[:, [0, 3, 6]] = 1
    mask = nd.array(mask.reshape((1, 3, 9)))
    mask = nd.topk(mask, axis=-1, k=3).sort().asnumpy().astype(int)
    img = nd.random.uniform(10, 20, shape=(2, 3, 12, 12))
    # target = img.reshape(6, 10, 10).transpose((1, 2, 0))
    # target = mx.image.resize_short(target, 5).transpose((2, 0, 1)).reshape(2, 3, 5, 5)
    # target = nd.tile(target, (1, 16, 1, 1))
    target = nd.ones(3)

    img = img.as_in_context(ctx)
    target = target.as_in_context(ctx)

    # test(ctx)
    test2()
    print 'ok'
