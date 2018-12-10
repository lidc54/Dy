import mxnet, os, math, random, sys
from mxnet import nd
from IN_data import loaders
import mxnet.gluon.data as data
from mxnet.gluon.data.vision import transforms

sys.path.append('../layers')
from params import global_param, alpha, c_rate  # using in initMX & loss_kernel
from dy_conv import constrain_kernal_num

# https://gluon-cv.mxnet.io/build/examples_classification/demo_imagenet.html#next-step
transform = transforms.Compose([transforms.RandomFlipLeftRight(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
crop_width, crop_height = 224, 224


def init_sphere(mnet, loaded_model=None, ctx=mxnet.cpu()):
    for k, v in mnet.collect_params().items():
        if 'bias' in k:
            v.initialize(mxnet.initializer.Constant(0.0), ctx=ctx)
        elif 'batchnorm' in k:
            if 'gamma' in k or 'var' in k:
                v.initialize(mxnet.initializer.Constant(1.0), ctx=ctx)
            elif 'beta' in k or 'mean' in k:
                v.initialize(mxnet.initializer.Constant(0.0), ctx=ctx)
        else:
            v.initialize(mxnet.initializer.Xavier(magnitude=3), ctx=ctx)
    print ':'
    print ':'
    if loaded_model != None:
        data = nd.load(loaded_model)
        for k, v in data.items():
            print k  # ,v.shape

    # if os.path.exists(loaded_model):
    # mnet.load_parameters(loaded_model, ctx=ctx, allow_missing=True)
    # gammas = {k: v for k, v in mnet.collect_params().items() if 'gamma' in k}
    # return gammas


def special_init(net, ctx=mxnet.cpu()):
    """special for 'vgg16-0000.params'"""
    loaded_model = 'vgg16-0000.params'
    key = ['arg:conv1_1_weight', 'arg:conv1_1_bias', 'arg:conv1_2_weight', 'arg:conv1_2_bias', 'arg:conv2_1_weight',
           'arg:conv2_1_bias', 'arg:conv2_2_weight', 'arg:conv2_2_bias', 'arg:conv3_1_weight', 'arg:conv3_1_bias',
           'arg:conv3_2_weight', 'arg:conv3_2_bias', 'arg:conv3_3_weight', 'arg:conv3_3_bias', 'arg:conv4_1_weight',
           'arg:conv4_1_bias', 'arg:conv4_2_weight', 'arg:conv4_2_bias', 'arg:conv4_3_weight', 'arg:conv4_3_bias',
           'arg:conv5_1_weight', 'arg:conv5_1_bias', 'arg:conv5_2_weight', 'arg:conv5_2_bias', 'arg:conv5_3_weight',
           'arg:conv5_3_bias', 'arg:fc6_weight', 'arg:fc6_bias', 'arg:fc7_weight', 'arg:fc7_bias', 'arg:fc8_weight',
           'arg:fc8_bias']
    loaded = nd.load(loaded_model)
    params = net.net.collect_params()
    for i, k in enumerate(params.keys()):
        name = key[i]
        params[k]._load_init(loaded[name], ctx)


def special_initMX(net, ctx=mxnet.cpu(), skip=(0, 0), layers=0):
    """special for 'vgg16-0000.params'
        skip: (start:end) the range where to skip when to init
        alert: this is for the one without BN layer
    """
    loaded_model = "/home1/caffemodel/VGG16.mnt"
    key = ['features.0.weight', 'features.0.bias', 'features.2.weight', 'features.2.bias', 'features.5.weight',
           'features.5.bias', 'features.7.weight', 'features.7.bias', 'features.10.weight', 'features.10.bias',
           'features.12.weight', 'features.12.bias', 'features.14.weight', 'features.14.bias', 'features.17.weight',
           'features.17.bias', 'features.19.weight', 'features.19.bias', 'features.21.weight', 'features.21.bias',
           'features.24.weight', 'features.24.bias', 'features.26.weight', 'features.26.bias', 'features.28.weight',
           'features.28.bias', 'features.31.weight', 'features.31.bias', 'features.33.weight', 'features.33.bias',
           'output.weight', 'output.bias']
    loaded = nd.load(loaded_model)
    params = net.net.collect_params()
    start, end = skip
    key[start:end] = []
    this_keys = params.keys()
    this_end = start + layers * 2
    key_weight = []
    if layers > 0:
        for k in this_keys[start:this_end]:
            if 'weight' in k:
                params[k].initialize(mxnet.initializer.Xavier(magnitude=3), ctx=ctx)
                key_weight.append(k)
            else:
                params[k].initialize(mxnet.initializer.Constant(0.0), ctx=ctx)
        # using compressed convolution in layers
        global_param.set_param(key_weight, ctx=ctx)
        this_keys[start:this_end] = []
    # print 'global_param.netMask.keys: ', global_param.netMask.keys()
    for name, k in zip(key, this_keys):
        params[k]._load_init(loaded[name], ctx)


def load_data(batch_size=64, num_workers=8, train_valid_ratio=0.8,
              list_path="/home1/ImageNet_ILSVRC2012/train_label.txt",
              data_path="/home1/ImageNet_ILSVRC2012/ILSVRC2012_img_train/"):
    data_loader = loaders(list_path, data_path)
    train_loader = data.DataLoader(
        data_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    num_train = len(data_loader)
    num_ratio = int(math.floor(train_valid_ratio * num_train))

    train_sampler = SubsetRandomSampler(num_train, num_ratio)
    valid_sampler = SubsetRandomSampler(num_train, num_train - num_ratio, train_sampler)

    train_loader = data.DataLoader(
        data_loader, batch_size=batch_size,
        sampler=train_sampler, last_batch='discard', num_workers=num_workers)
    valid_loader = data.DataLoader(
        data_loader, batch_size=batch_size,
        sampler=valid_sampler, last_batch='discard', num_workers=num_workers)

    return train_loader, valid_loader


def loss_kernel(net, ctx):
    '''loss for weight kernel in net'''
    losses = []
    for i_c, c in zip(range(len(ctx)), ctx):
        this_ctx = []
        for k_net in global_param.selected_key['group'][i_c]:
            kk = global_param.selected_key[k_net]
            this_ctx.append((kk, net.collect_params()[kk]._data[i_c]))
        param = dict(this_ctx)
        loss = constrain_kernal_num(param, ctx=c)
        losses.append(loss * alpha)
    return losses


def lossa_compress(net, axis):
    """compress the weight of net: gradiant: group
        return log & loss"""
    params = net.collect_params()
    res = []
    log_res = {}
    threshold = 1e-6
    num_ctx = 0
    for key, value in params.items():
        if 'weight' not in key: continue
        ctx = value.list_ctx()
        num_ctx = len(ctx)
        for c in ctx:
            val = value.data(ctx=c)
            res.append(nd.sum(nd.sqrt(nd.sum(val ** 2, axis))))
            if key + '_Line' in log_res.keys(): continue
            log_res[key + '_Col'] = nd.sum(nd.sum(val, axis=0) < threshold).asscalar() \
                                    / reduce(lambda x, y: x * y, nd.sum(val, axis=0).shape)
            log_res[key + '_Line'] = nd.sum(nd.sum(val, axis=1) < threshold).asscalar() \
                                     / reduce(lambda x, y: x * y, nd.sum(val, axis=1).shape)
    res1 = [nd.sum(nd.stack(*res[i::num_ctx])) for i in range(num_ctx)]
    return res1, log_res


class SubsetRandomSampler():
    """
    exclude is a instance of SubsetRandomSampler
    """

    def __init__(self, length, subs, exclude=None):
        self._length = length
        self._subs = subs
        self.exclude = exclude

    def __iter__(self):
        random.seed(random.randint(1, 100))
        indices = range(self._length)
        if self.exclude:
            try:
                indices = set(indices) - set(self.exclude.out)
            except Exception, e:
                pass
            self.exclude = None
        self.out = random.sample(indices, self._subs)
        return iter(self.out)

    def __len__(self):
        return self._subs


def mean_std(v):
    sum = nd.sum(v)
    n_count = nd.sum(v != 0)
    mean = sum / n_count
    std = nd.sum(v ** 2) - n_count * (mean ** 2)
    std = nd.sqrt(std / n_count)
    return mean, std


def online_check(epoch, sw, paramer):
    '''using globalparams'''
    # 1. get key
    keys = global_param.netMask.keys()
    new_keys = []
    for k in keys:
        if '_muX' in k or '_stdX' in k:
            continue
        else:
            new_keys.append(k)
    # 2. get data of mask
    try:
        for k in new_keys:
            data = global_param.netMask[k][:]
            _, _, k1, k2 = data.shape
            for i in range(2):
                data = nd.sum(data, axis=-1)
            tag = k + '___' + repr(k1 * k2)
            sw.add_histogram(tag=tag, values=data, global_step=epoch, bins=k1 * k2)
    except Exception, e:
        print e, 'there is a netMask here!'
        # pass
    # 3. get data from net
    try:
        selected = global_param.selected_key['total']
        for k in selected:
            data = paramer[k]._data[0]
            _, _, k1, k2 = data.shape
            mu, std = mean_std(data)
            thr = mu + c_rate * std
            out = data > thr  # base to 0
            for i in range(2):
                out = nd.sum(out, axis=-1)
            tag = k + '___' + repr(k1 * k2)
            sw.add_histogram(tag=tag, values=out, global_step=epoch, bins=k1 * k2)
    except Exception, e:
        print e, 'no total'
