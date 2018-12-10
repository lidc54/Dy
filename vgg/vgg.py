# from layers.dy_conv import origin_conv, new_conv
from unit import init_sphere, special_init
from mxnet.gluon import nn, HybridBlock
import sys, os

sys.path.append('../layers')
from dy_conv import new_conv
from params import gls


class vgg(nn.Block):  # HybridBlock
    def __init__(self, use_bn=True, **kwargs):
        super(vgg, self).__init__(**kwargs)
        self.build_vgg()

    def forward(self, x):
        return self.net(x)

    # def hybrid_forward(self, F, x, *args, **kwargs):
    #     x = self.net(x)
    #     return super(vgg, self).hybrid_forward(F, x, *args, **kwargs)

    def vgg_block(self, num_convs, channels, in_channels, kernel_size=3, my_fun=nn.Conv2D):
        out = nn.HybridSequential()
        # todo: here change to different covolutional function
        if gls.switch_conv and kernel_size != 3:  # change convolution mode here
            my_fun = new_conv
        for _ in range(num_convs):
            conv = my_fun(channels, in_channels=in_channels[0],
                          kernel_size=kernel_size, padding=int(kernel_size / 2))
            in_channels[0] = channels
            relus = nn.Activation('relu')
            # bn=nn.BatchNorm()
            out.add(conv, relus)
        out.add(nn.MaxPool2D(pool_size=2, strides=2))
        return out

    def vgg_stack(self, architecture):
        out = nn.HybridSequential()
        in_channels = [3]
        # my_fun = new_conv
        # nn_func = nn.Conv2D
        for arch in architecture:
            if len(arch) == 2:
                num_convs, channels = arch
                kernel_size = 3
                # function = nn_func
            else:
                num_convs, channels, kernel_size = arch
                # function = my_fun
            out.add(self.vgg_block(num_convs, channels, in_channels,
                                   kernel_size=kernel_size))
        return out

    def build_vgg(self):
        num_outputs = 1000
        # architecture = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
        # architecture = ((1, 64, 5), (1, 128, 5), (3, 256), (3, 512), (3, 512))
        # move to params in gls
        architecture = gls.arch
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(self.vgg_stack(architecture=architecture))
            self.net.add(nn.Flatten())
            self.net.add(nn.Dense(4096, activation="relu"))
            self.net.add(nn.Dropout(.5))
            self.net.add(nn.Dense(4096, activation="relu"))
            self.net.add(nn.Dropout(.5))
            self.net.add(nn.Dense(num_outputs))


#############################################################################################
#                                                                                           #
#############################################################################################
class compress_vgg(nn.Block):
    """
        for build compressed net, a new idea: replace small kernel with bigger
        check detail in
    """

    def __init__(self, my_fun=nn.Conv2D, use_bn=True):
        super(compress_vgg, self).__init__()
        self.build_vgg(my_fun)

    def forward(self, x):
        return self.net(x)

    def build_vgg(self, func):
        num_outputs = 1000
        architecture = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
        self.net = nn.Sequential()
        with self.net.name_scope():
            self.net.add(self.vgg_stack(architecture=architecture, function=func))
            self.net.add(nn.Flatten())
            self.net.add(nn.Dense(4096, activation="relu"))
            self.net.add(nn.Dropout(.5))
            self.net.add(nn.Dense(4096, activation="relu"))
            self.net.add(nn.Dropout(.5))
            self.net.add(nn.Dense(num_outputs))

    def vgg_stack(self, architecture, function=nn.Conv2D):
        out = nn.Sequential()
        in_channels = [3]
        trans = {2: 5, 3: 7}  # used for inference bigger kernel
        for (num_convs, channels) in architecture:
            kernel_sz = trans[num_convs]
            padding = kernel_sz / 2
            conv = function(channels, in_channels=in_channels[0],
                            kernel_size=kernel_sz, padding=padding)
            relus = nn.Activation('relu')
            in_channels[0] = channels
            out.add(conv, relus)
        return out


if __name__ == "__main__":
    # net = vgg()
    net = vgg()  # compress_vgg()
    model = 'vgg16-0000.params'
    special_init(net)
    # print net.net.collect_params().keys()
    for key, value in net.net.collect_params().items():
        print key, value.shape
    # init_sphere(net,model)
