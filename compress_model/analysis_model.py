'''
analysis the model:
1. extract import paramers
2. get their index of input and output
3. analysis the nums equals 3 & ratio of is and isnot 3

-**- so, one can get paramers & location directly from *.model -**-
'''
from mxnet import nd
import pickle, os, mxnet

name_pre, name_aft = ['spherenet20', 'spherenet20_3']


def analy_model(mask=None, model=None, kernel_size=(1, 3), show=False):
    '''
    for build an curve of numbers of paramers in kernel
    and
    extract top 3 paramers; got related mask; obtain the right order of key
    '''
    from layers.dy_conv import new_conv
    from units import init_sphere
    from layers.sphere_net import SphereNet20
    if mask is None:
        mask = "/home/ldc/PycharmProjects/Dy/log_4dy_Ns3/global.param"
    if model is None:
        model = 'log_4dy_Ns3/spherenet_ft_Ns.model'
    ctx = mxnet.cpu()

    mnet = SphereNet20(my_fun=new_conv, use_bn=False)
    # gammas = init_sphere(mnet, model, ctx)
    # paramers = nd.load(model)
    netMask = {}
    if os.path.exists(mask):
        with open(mask)as f:
            sv = pickle.load(f)
            for k, v in sv.netMask.items():
                netMask[k] = v.as_in_context(ctx)
    all = 0
    static = {}
    paramers = {}
    cal_mask = {}
    loaded = nd.load(model)
    k = loaded.keys()
    keyorder = mnet.collect_params().keys()
    loaded_key = rearrange(target_key=keyorder, needfix_key=k,show=show)
    for idx_key, key in enumerate(keyorder):
        t_k = loaded_key[idx_key]
        value = loaded[t_k]
        if not ('conv' in key and 'weight' in key):
            paramers[key] = value
            continue
        size = value.shape
        output, input = size[:2]
        # name = 'spherenet200_' + '_'.join(key.split('.')[2:])
        ISname = key in netMask.keys()
        masked = netMask[key]
        # print key, ':',
        masked = masked.reshape(size[:2] + (-1,))
        masked = nd.sum(masked, axis=-1)
        static[key] = nd.zeros(output)
        static[key + '_minus'] = nd.zeros(output)
        if all < output: all = output
        for i in range(output):
            static[key][i] = nd.sum(masked[i] > 3) / input
            static[key + '_minus'][i] = nd.sum(masked[i] < 3) / input
        static[key] = static[key].sort()
        static[key + '_minus'] = static[key + '_minus'].sort()
        # --------------------deal with paramers in net----------------
        N, C, K1, K2 = value.shape
        value_trans = value.reshape(N, C, -1)
        tops_mask = nd.topk(nd.abs(value_trans), k=3, ret_typ='mask')
        tops_idx = nd.topk(tops_mask, k=3, ret_typ='indices')
        value_trans = value_trans.reshape(-1, K1 * K2)
        cal_mask[key] = tops_idx.asnumpy().astype(int)
        tops_idx = tops_idx.reshape(-1, 3)

        out = []
        for x in range(3): out.append(value_trans[range(N * C), tops_idx[:, x]])
        paramers[key] = nd.stack(*out).transpose().reshape((N, C) + kernel_size)
        print key
    print 'analysis loop stop'
    if show:
        from mxboard import SummaryWriter
        sw = SummaryWriter(logdir='sphere_dynamic', flush_secs=20000)
        for j in range(all):
            for k, v in static.items():
                if j >= v.shape[0]:
                    continue
                sw.add_scalar(tag=k, value=v[j].asscalar(), global_step=j)

    return paramers, cal_mask, keyorder


############################################################################
def rearrange(target_key=None, needfix_key=None, show=True):
    '''
    check the difference between new and old network
    rearrange them in same order
    '''
    if target_key is None or needfix_key is None:
        from layers.params import prefix, collect
        needfix_key, target_key = prefix, collect

    def mykey(c):
        r = c.split('.')
        maps = {'weight': 1, 'bias': 2, 'alpha': 3, 'gamma': 4, 'beta': 5, 'running_mean': 6, 'running_var': 7}
        if len(r) == 2:
            key = 2000
        elif len(r) == 3:
            key = 1000
        else:
            key = 100 * (1 + int(r[1])) + 10 * int(r[2][-1])
        last = maps[r[-1]]
        return key + last

    def cmp(a, b):
        return a - b

    p = sorted(needfix_key, cmp=cmp, key=mykey)
    if show:
        print ''
        for a, b in zip(p, target_key):
            print '1 %-25s%-35s' % (a, b)
    return p


def transfer_load(exist_model=None, ctx=mxnet.cpu()):
    '''
    the aim of this function is and only is
    to transfer compressed model paramers to a new model
    '''
    from general_conv import SphereNet20_3
    # from layers.sphere_net import SphereNet20
    net = SphereNet20_3()
    print  net.__dict__.keys()

    t_params, m, t_k = analy_model(model=exist_model)
    n_params = net._collect_params_with_prefix()
    # t_params = target_net._collect_params_with_prefix()
    n_k = n_params.keys()
    # t_k = t_params.keys()
    re_n_k = rearrange(needfix_key=n_k, target_key=t_k, show=False)
    # n_k=net.collect_params().keys()
    # t_k=target_net.collect_params().keys()
    for origin_name, target_name in zip(re_n_k, t_k):
        # origin_value = net.collect_params()[origin_name]
        # target_value = target_net.collect_params()[target_name]
        # origin_value.set_data(target_value.data().copy())
        '''
        alert: why paramer.data()[ndarray] cannot used in _load_init,
        and paramer[key]._reduce() is also a ndarray
        because the later has function copyto() used in _init_impl()
        '''
        n_params[origin_name]._load_init(t_params[target_name], ctx)
        print '2', origin_name, target_name
    print 'loop stop'
    # save
    _path = list(os.path.split(exist_model))
    _path[-1:-1] = ['transfed']
    model_path = _path[:]
    mask_path = _path[:]
    mask_path[-1] = 'global.param'
    base_path = os.path.join(*model_path[:-1])
    model_path = os.path.join(*model_path)
    mask_path = os.path.join(*mask_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    net.save_params(model_path)

    # change name
    cal_mask = {}
    for k, v in m.items():
        k_new = k.replace(name_pre, name_aft)
        cal_mask[k_new] = v
    with open(mask_path, 'w')as f:
        pickle.dump(cal_mask, f)
    print 'path for model: ', model_path
    print 'path for masks: ', mask_path


if __name__ == "__main__":
    # p, m, k = analy_model()
    # rearrange() 
    transfer_load("/home/ldc/PycharmProjects/Dy/log_4dy_Ns3/spherenet_ft_Ns.model")  #
