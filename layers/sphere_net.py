import mxnet.gluon.nn as nn
import mxnet.gluon as gluon
from mxnet import nd
import mxnet as mx
# import gradcam, pickle, dataset
import math
from layers import data
from layers.dy_conv import origin_conv


# from trans_model_paramers import myinit

# the base block for shpereface; notice the bn layer
class Residual(nn.Block):
    def __init__(self, channels, in_channels, same_shape=True, my_fun=nn.Conv2D, use_bn=False, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        self.use_bn = use_bn
        if not same_shape:
            self.conv0 = my_fun(channels, in_channels=in_channels[0],
                                kernel_size=3, padding=1, strides=2)
            self.a0 = mPReLU(channels)
            if self.use_bn: self.b0 = nn.BatchNorm()
            in_channels[0] = channels
        self.conv1 = my_fun(channels, kernel_size=3,
                            in_channels=in_channels[0], padding=1)
        self.a1 = mPReLU(channels)
        if self.use_bn: self.b1 = nn.BatchNorm()
        in_channels[0] = channels
        self.conv2 = my_fun(channels, kernel_size=3,
                            in_channels=in_channels[0], padding=1)
        self.a2 = mPReLU(channels)
        if self.use_bn: self.b2 = nn.BatchNorm()
        in_channels[0] = channels

    def forward(self, x):
        if not self.same_shape:
            x = self.a0(self.conv0(x))
            if self.use_bn: x = self.b0(x)
        out = self.a1(self.conv1(x))
        if self.use_bn: out = self.b1(out)
        out = self.a2(self.conv2(out))
        if self.use_bn: out = self.b2(out)
        return out + x


class AngleLinear(mx.gluon.HybridBlock):
    def __init__(self, in_features, out_features, m=4, weight_initializer=None):
        super(AngleLinear, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        self.weight = self.params.get('weight', shape=[out_features, in_features],
                                      init=weight_initializer,
                                      allow_deferred_init=True)
        # self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        # self.phiflag = phiflag
        self.m = m
        # cos(0x);cos(x);...;cos(4X);cos(5x)--x:cos(x)
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def hybrid_forward(self, F, x, *args, **params):
        # xsize=(B,F)    F is feature len
        w = self.weight._data[0]  # size=(Classnum,F) F=in_features Classnum=out_features

        ww = nd.L2Normalization(w)
        xlen = x.square().sum(axis=1, keepdims=True).sqrt()
        wlen = ww.square().sum(axis=1, keepdims=True).sqrt()

        cos_theta = nd.dot(x, ww.T) / xlen.reshape(-1, 1) / wlen.reshape(1, -1)
        cos_theta = cos_theta.clip(-1, 1)

        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = nd.arccos(cos_theta)
        k = (self.m * theta / math.pi).floor()
        phi_theta = (-1 ** k) * cos_m_theta - 2 * k

        cos_theta = cos_theta * xlen.reshape(-1, 1)
        phi_theta = phi_theta * xlen.reshape(-1, 1)
        output = (cos_theta, phi_theta)
        return output  # size=(B,Classnum,2)


class AngleLoss(gluon.loss.Loss):
    def __init__(self, weight=1, batch_axis=0, gamma=0, **kwargs):
        super(AngleLoss, self).__init__(weight, batch_axis, **kwargs)
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def hybrid_forward(self, F, xcos_theta, xphi_theta, target):
        self.it += 1
        batch_size = target.size  # size = (B,classnum)
        oh_target = target.one_hot(xcos_theta.shape[1])
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        # because indexing is not differentiable in mxnet, we must do this
        output = xcos_theta - \
                 oh_target * xcos_theta[range(0, batch_size), target].reshape(-1, 1) / (1 + self.lamb) + \
                 oh_target * xphi_theta[range(0, batch_size), target].reshape(-1, 1) / (1 + self.lamb)
        loss = nd.softmax_cross_entropy(output, nd.cast(target, 'float32'))  # (B,Classnum)
        return loss


def test_accur(target, it, *input):
    LambdaMin = 5.0
    LambdaMax = 1500.0
    lamb = 1500.0
    theta, phi = input
    batch_size = target.size
    lamb = max(LambdaMin, LambdaMax / (1 + 0.1 * it))
    # because indexing is not differentiable in mxnet, we must do this
    output = theta - theta / (1 + lamb) + phi / (1 + lamb)
    nd.softmax(output, out=output)
    v, idx = nd.topk(output, ret_typ='both')
    real = (idx == target.reshape(-1, 1).astype(idx.dtype))
    return nd.sum(real) / batch_size, nd.sum(real * v) / batch_size


class mPReLU(nn.HybridBlock):
    '''
    modify
     the official prelu
    '''

    def __init__(self, num_units, initializer=None, **kwargs):
        super(mPReLU, self).__init__(**kwargs)
        self.num_units = num_units
        with self.name_scope():
            self.alpha = self.params.get('alpha', shape=(1, num_units, 1, 1), init=initializer,
                                         grad_req='write')

    def hybrid_forward(self, F, x, alpha):
        return mx.nd.maximum(x, 0) + mx.nd.minimum(alpha * x, 0)


class SphereNet20(nn.Block):
    # http://ethereon.github.io/netscope/#/gist/20f6ddf70a35dec5019a539a502bccc5
    def __init__(self, num_classes=10574, verbose=False, my_fun=nn.Conv2D, use_bn=True, **kwargs):
        super(SphereNet20, self).__init__(**kwargs)
        self.verbose = verbose
        self.feature = True  # weather only fc1 be returned or train with a classifier
        # add name_scope on the outermost Sequential
        in_put = [3]
        with self.name_scope():
            # block 1
            self.net = nn.Sequential()
            b1 = Residual(64, in_put, my_fun=my_fun, use_bn=use_bn, same_shape=False)

            # block 2
            b2_1 = Residual(128, in_put, my_fun=my_fun, use_bn=use_bn, same_shape=False)
            b2_2 = Residual(128, in_put, my_fun=my_fun, use_bn=use_bn)

            # block3
            b3_1 = Residual(256, in_put, my_fun=my_fun, use_bn=use_bn, same_shape=False)
            b3_2 = Residual(256, in_put, my_fun=my_fun, use_bn=use_bn)
            b3_3 = Residual(256, in_put, my_fun=my_fun, use_bn=use_bn)
            b3_4 = Residual(256, in_put, my_fun=my_fun, use_bn=use_bn)

            # block 4
            b4 = Residual(512, in_put, my_fun=my_fun, use_bn=use_bn, same_shape=False)
            f5 = nn.Dense(512, in_units=21504)  #
            self.f6 = AngleLinear(512, num_classes)
            self.net.add(b1, b2_1, b2_2, b3_1, b3_2, b3_3, b3_4, b4, f5)

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


def init_model():
    mnet = SphereNet20()
    ctx = mx.gpu()
    loaded_model = ""
    if not loaded_model:
        for k, v in mnet.collect_params().items():
            if 'bias' in k:
                v.initialize(mx.initializer.Constant(0.0), ctx=ctx)
                continue
            v.initialize(mx.initializer.Xavier(magnitude=3), ctx=ctx)
    # mnet.initialize_(mx.gpu())
    train_data_loader, valid_data_loader \
        = data.train_valid_test_loader("/home1/CASIA-WebFace/aligned_Webface-112X96/", batch_size=16)
    mnet.feature = False
    for batch, label in valid_data_loader:
        batch = batch.as_in_context(ctx)
        out = mnet(batch)

        break

    mnet.save_params("./spherenet_model")
    return mnet


def load_model():
    mnet = SphereNet20()
    mnet.load_params("./spherenet_model")
    return mnet


def build_net(loaded_model, use_bn=True):
    from units import init_sphere
    my_fun = origin_conv
    mnet = SphereNet20(my_fun=my_fun, use_bn=use_bn)
    ctx = mx.cpu()
    gammas = init_sphere(mnet, loaded_model, ctx)
    return mnet, gammas


def mean_std(v):
    sum = nd.sum(v)
    n_count = nd.sum(v != 0)
    mean = sum / n_count
    std = nd.sum(v ** 2) - n_count * (mean ** 2)
    std = nd.sqrt(std / n_count)
    return mean, std


def check_gamma():
    loaded_model = "log_bn_dy/spherenet_bn_4dy.model"
    _, gammas = build_net(loaded_model)
    for k, v in gammas.items():
        mean, std = mean_std(v)
        print k, 'mean:%d, std:%d' % (mean.asscalar(), std.asscalar())


def check_kernel_nums(mnet, loaded_param=None, loaded_model=None):
    from layers.params import c_rate
    # loaded_model = "log_4dy_Ns2/spherenet_ft_Ns.model"
    # mnet, _ = build_net(loaded_model, use_bn=False)
    #
    # for k, v in mnet.collect_params().items():
    #     data = v.data()
    #     mean, std = mean_std(data)
    #     print k, 'mean--- %.5f---,with the shape std---%.5f---with the shape ' % (
    #         mean.asscalar(), std.asscalar()), v.shape
    exclude = ['alpha', 'bias', 'dense', '_weight_']  # 'conv0_', 'conv1_', 'conv2_',
    if loaded_param is None:
        loaded_param = "../sphere_dynamic/log_4dy_Ns3/global.param"
    import pickle, math
    with open(loaded_param)as f:
        dd = pickle.load(f)
    pramer = dd.netMask
    # print 'iter:', dd.iter
    import matplotlib.pyplot as plt
    cout, cols, idx = 0.0, 2, 0
    next = False
    for k in pramer.keys():
        pramer[k] = pramer[k].as_in_context(mx.cpu())
        for x in exclude:
            if x in k:
                next = True
                break
        if next:
            next = False
            continue
        if 'weight' in k:
            cout += 1
    rows = int(math.ceil(cout / cols))
    print 'total:', cout, 'rows:', rows, 'cols:', cols
    keyorder = pramer.keys()
    lm = False
    if not loaded_model is None:
        from compress_model.analysis_model import rearrange
        # from layers. import SphereNet20
        from dy_conv import new_conv, origin_conv
        lm = True
        # mnet = SphereNet20(my_fun=origin_conv, use_bn=False)
        keyorder = mnet.collect_params().keys()
        loaded = nd.load(loaded_model)
        re_n_k = rearrange(needfix_key=loaded.keys(), target_key=keyorder, show=False)
    next = False
    for j, k in enumerate(keyorder):
        try:
            v = pramer[k]
        except Exception, e:
            print e
            continue
        for x in exclude:
            if x in k:
                next = True
                break
        if 'weight' not in k or next:
            next = False
            continue
        tag = k.split('_')[1]
        idx += 1
        _, _, k, _ = v.shape
        out = v
        if lm:
            try:
                mu, std = pramer[k + '_muX'], pramer[k + '_stdX']
            except Exception, e:
                mu, std = mean_std(v)

            thr = (mu + c_rate * std)
            # todo:mask
            out2 = nd.abs(loaded[re_n_k[j]]) / thr - 1
            temp = nd.zeros_like(out2)
            out2 = nd.where(nd.abs(out2) > 0.1, temp, out2)

        for i in range(2):
            if lm: out2 = nd.sum(out2, axis=-1)
            out = nd.sum(out, axis=-1)
        ax = plt.subplot(rows, cols, idx)

        # ratio of 1/3 [numbers in kernel]
        ratio1 = nd.sum(nd.where(out >= 2, nd.where(out <= 4, nd.ones_like(out),
                                                    nd.zeros_like(out)), nd.zeros_like(out)))
        ratio = (ratio1 / nd.sum(nd.ones_like(out))).asscalar()

        # plot
        out = nd.sort(out.reshape(-1)).asnumpy()
        x = range(len(out))
        l1, = ax.plot(x, out, 'r', label=tag + '_' + repr(ratio))
        print 'len kenels', tag, len(out)
        if lm:
            out2 = out2 * 10 + 15
            out2 = nd.sort(out2.reshape(-1)).asnumpy()
            l2, = ax.plot(x, out2, 'b', label='X20plus5')
        ax.set_yticks(range(1, 1 + k * k))
        if lm:
            ax.legend([l1, l2], loc='best')
        else:
            ax.legend(loc='best')

        # histogram
        # out = out.reshape(-1).asnumpy()
        # ax.hist(out, bins=27, label=tag)
        # ax.set_xticks(range(9))
    plt.savefig('plotDis.jpg')
    plt.show()

    print('o')


if __name__ == "__main__":
    # mnet = init_model()
    # load_model()
    loaded_param = "../sphere_dynamic/log_4dy_Ns3/global.param"
    check_kernel_nums(loaded_param)
