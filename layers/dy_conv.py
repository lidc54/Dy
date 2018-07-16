from mxnet.gluon import nn
from mxnet import nd, initializer, autograd
from mxnet.gluon import loss, Trainer
import mxnet as mx
import math, random
from params import *


# from mxboard import *
# sw = SummaryWriter(logdir='/home/ldc/data/logs', flush_secs=5)


class origin_conv(nn.Conv2D):
    def __init__(self, channels, kernel_size, **kwargs):
        super(origin_conv, self).__init__(channels, kernel_size, **kwargs)


class new_conv(nn.Conv2D):
    def __init__(self, channels, kernel_size, **kwargs):
        # if isinstance(kernel_size, base.numeric_types):
        #     kernel_size = (kernel_size,)*2
        # assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"

        super(new_conv, self).__init__(channels, kernel_size, **kwargs)

    # def  forward(self, x, *args):
    #     self.ctx = x.context
    #     self.set_params()
    #     return super(new_conv, self).forward(x, *args)
    def hybrid_forward(self, F, x, weight, bias=None):
        keys = self.params.keys()
        # self.assign_mask(keys)
        for key in keys:
            if 'weight' in key:
                wmask = global_param.netMask[key] = \
                    assign_mask(weight, global_param.netMask[key], key)
            else:
                bmask = global_param.netMask[key] = \
                    assign_mask(bias, global_param.netMask[key])
        return super(new_conv, self).hybrid_forward(F, x, weight * wmask, bias * bmask)


# forward with mask
def assign_mask(weight, mask, key=None):
    # Calculate the mean and standard deviation of learnable parameters
    # maskshape = weight.shape
    # nd.sigmoid(weight),
    iter_ = global_param.iter
    masked = weight * mask
    mu = nd.sum(nd.abs(masked))
    std = nd.sum(masked * weight)
    count = nd.sum(masked != 0)
    all_count = reduce(lambda x, y: x * y, masked.shape)
    mu = mu / count
    std = std - count * nd.power(mu, 2)
    std = nd.sqrt(std / count)
    mu = mu.asscalar()
    std = std.asscalar()
    if key:
        tag_key = '_'.join(key.split('_')[1:])
        tag_shape = reduce(lambda x, y: x + 'X' + y, map(str, masked.shape))
        tag = [tag_key, tag_shape, str(all_count)]
        tag = '_'.join(tag)
        value = 1.0 * count.asscalar() / all_count
        sw.add_scalar(tag=tag, value=value, global_step=global_param.iter)
    # Calculate the weight mask and bias mask with probability
    r = random.random()
    if math.pow(1 + gamma * iter_, -power) > r and iter_ < iter_stop:
        cond1 = (mask == 1) * (nd.abs(weight) < (0.9 * max(mu + c_rate * std, 0)))
        cond2 = (mask == 0) * (nd.abs(weight) > (1.1 * max(mu + c_rate * std, 0)))
        mask = mask - cond1 + cond2
    return mask


def constrain_kernal_num():
    # L1_loss = loss.L1Loss()
    exclude = ['conv0_', 'conv1_', 'conv2_', 'alpha', 'bias', 'dense']

    def Not_excluded(k):
        for i in exclude:
            if i in k:
                return False
        return True

    num_kernel = [constrain_conv(v, k) for k, v in global_param.netMask.items() if Not_excluded(k)]
    loss_nums = reduce(lambda x, y: x + y, num_kernel)
    # Calculate the weight for mask
    iter_ = global_param.iter
    r = 1.0 - math.pow(1 + advanced_iter * gamma * iter_, -power)
    return loss_nums * r * nums_power


# a convlution with constrain of limited paramers
def constrain_conv(mask, name):
    out = mask
    for i in range(2): out = nd.sum(out, axis=-1)
    channel = out.shape[1]
    out1 = nd.abs(out - kept_in_kernel)  # close to kept_in_kernel,for example 3;
    out2 = nd.abs(out)  # close t0 0
    out_c = nd.where(out1 < out2, out1, out2)
    w = out_c / nd.sum(out_c)  # distribution the derivative
    constrain = nd.sum(w * out_c) / channel
    tag_key = 'K_' + '_'.join(name.split('_')[1:])
    sw.add_scalar(tag=tag_key, value=constrain.asscalar(), global_step=global_param.iter)
    return constrain


# gamma in BN will be used for dynamic compress in a structure way.
# the method is discarded.we will compress relative items where mask masked.
class new_BN(nn.BatchNorm):
    def __init__(self):
        super(new_BN, self).__init__()

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        key = self.name + '_gamma'
        mask = global_param.netMask[key] = \
            assign_mask(gamma, global_param.netMask[key], key)
        return super(new_BN, self).hybrid_forward(F, x, gamma * mask, beta, running_mean, running_var)


if __name__ == "__main__":
    a = nd.random.uniform(0, 1, shape=(1, 1, 5, 5))
    lr = 0.1
    net = nn.Sequential()

    cov = new_conv(2, 3, in_channels=1, padding=1, strides=1)
    with net.name_scope():
        net.add(new_conv(2, 3, in_channels=1, padding=1, strides=1))
        net.add(new_conv(1, 3, in_channels=2, strides=1, padding=1))

    ctx = mx.cpu()
    L = loss.L1Loss()
    net.collect_params().initialize(init=initializer.Xavier(magnitude=3), ctx=ctx)
    trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': 0.0})
    key = net.collect_params().keys()
    a = a.as_in_context(ctx)

    global_param.set_param(key, ctx=ctx)

    for i in range(15000):
        with autograd.record():
            global_param.iter = i
            out = net[0](a)
            out = net[1](out)
            loss1 = L(out, a)
            print(loss1.asscalar())
        loss1.backward()
        sw.add_scalar(tag='loss', value=loss1.asscalar(), global_step=i)
        trainer.step(1)
        if i != 0 and i % 1000 == 0:
            lr *= 0.5
            trainer.set_learning_rate(lr)
    print('ok')
