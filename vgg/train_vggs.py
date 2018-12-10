from vgg import vgg
from unit import *  # special_init, load_data,transform
from mxboard import SummaryWriter
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.io import ImageRecordIter
import os, pickle, time
from params import gls, global_param


def train_vgg(gpu=0, lr=0.001, root='', param_file='', Isinit=True, FLF=False):
    num = 1
    batch_size = 40 * num  # 32
    num_workers = 2 * num
    lr = lr
    epoch = 8
    ratio_train_test = 0.8

    ctx = [mx.gpu(i) for i in [gpu]]  # range(8 - gpu, 8)
    net = vgg()

    # net.collect_params().initialize(ctx=ctx)
    if Isinit:
        # special_init(net, ctx=ctx)
        special_initMX(net, ctx=ctx, skip=gls.initp[0], layers=gls.initp[1])
        file = os.path.join(root, 'vgg.models')
    else:
        file = os.path.join(root, param_file)
        net.load_parameters(file, ctx=ctx)
    loaded_param = file.replace('.model', '_mask.pkl')
    global_param.load_param(loaded_param, ctx)
    # net.hybridize()

    # train_data, valid_data = load_data(batch_size=batch_size, num_workers=num_workers)
    data_iter = ImageRecordIter(batch_size=batch_size, data_shape=(3, crop_width, crop_height), shuffle=True,
                                path_imgrec="/home1/ImageNet_ILSVRC2012/train_label.rec",
                                path_imgidx="/home1/ImageNet_ILSVRC2012/train_label.idx",
                                aug_list=transform, preprocess_threads=num_workers)
    # params = net.net.collect_params()
    params = {}
    # try:
    #    for i in global_param.selected_key['total']:
    #        params[i]=net.net.collect_params()[i]
    # except Exception,e:
    for i in range(gls.initp[1]):
        pkey = net.net.collect_params().keys()[i]
        params[pkey] = net.net.collect_params()[pkey]
    trainer = gluon.Trainer(params, 'adam', {'learning_rate': lr})

    CEloss = gluon.loss.SoftmaxCrossEntropyLoss(axis=-1, sparse_label=True)

    epoch_train = 0  # total records
    valid = 0
    for epochs in range(epoch):
        j = epochs * epoch_train
        t = time.time()
        i = 0
        for contain in data_iter:
            i += 1
            global_param.iter = i + j
            batch, label = (contain.data[0], contain.label[0])
            # batch = batch.as_in_context(ctx)
            # label = label.as_in_context(ctx)
            batch = gluon.utils.split_and_load(batch, ctx)
            label = gluon.utils.split_and_load(label, ctx)
            if i < ratio_train_test * epoch_train or epoch_train == 0:
                with autograd.record():
                    losses = [CEloss(net(X), Y) for X, Y in zip(batch, label)]
                    losses = [mx.nd.sum(X) for X in losses]
                    # todo:1.loss;2.init
                    if FLF:
                        loss_k = loss_kernel(net, ctx)
                        # lossa, log_lossa = lossa_compress(net, 1)#loss for net structure
                        loss_all = [X + Y for X, Y in zip(losses, loss_k)]
                    else:
                        loss_all = losses
                for loss in loss_all:
                    loss.backward()
                trainer.step(batch_size)
                value = [X.asscalar() for X in losses]
                value = reduce(lambda X, Y: X + Y, value) / batch_size
                sw.add_scalar(tag='Loss', value=value, global_step=i + j)
                # for k_sw, v_sw in log_lossa.items():
                #     sw.add_scalar(tag=k_sw, value=v_sw, global_step=i + j)
                if i % 200 == 0:
                    print('iter:%d,loss:%4.5f,time:%4.5fs' % (i + j, value, time.time() - t))
                    t = time.time()

            else:
                out = [net(X) for X in batch]
                # value1, idices1 = mx.nd.topk(out, ret_typ='both')
                out = [mx.nd.softmax(X, axis=1) for X in out]
                tops = [(mx.nd.topk(X, ret_typ='both'), Y)
                        for X, Y in zip(out, label)]
                # print mx.nd.sum(value == value1).asscalar(), mx.nd.sum(idices == idices1).asscalar()
                disc, cont = 0, 0
                for (value_, idices_), label_ in tops:
                    real = idices_.reshape(-1) == label_.astype(idices_.dtype)
                    disc += mx.nd.sum(real).asscalar()
                    cont += mx.nd.sum(real * value_.T).asscalar()

                # for a, b in zip(label.asnumpy().astype(np.uint),
                #                 idices.reshape(-1).asnumpy().astype(np.uint)):
                #     if not a in test.keys():
                #         test[a]=set([])
                #     test[a]=set(list(test[a]).append(b))

                discroc = disc / batch_size  # (mx.nd.sum(real) / batch_size).asscalar()
                controc = cont / batch_size  # (mx.nd.sum(real * value) / batch_size).asscalar()
                sw.add_scalar(tag='RocDisc', value=discroc, global_step=valid)
                sw.add_scalar(tag='RocCont', value=controc, global_step=valid)
                valid += 1
                if i % 200 == 0:
                    print 'RocDisc', discroc

        data_iter.reset()
        if i > epoch_train: epoch_train = i
        print 'epcoah length:', epoch_train, 'and i:', i, 'time:', time.time() - t
        online_check(epochs, sw, net.collect_params())
        # save model
        net.save_parameters(file)
        global_param.save_param(loaded_param)
        print '*' * 30


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser(description='paramers for compressed model trainning')
    parse.add_argument('--gpuid', type=int, default=1, help='the No. of gpus, default=1')
    parse.add_argument('--root', type=str, default='vggs', help='root path to save & load model')
    parse.add_argument('--Isinit', type=bool, default=True, help='whether to initialize the net')
    parse.add_argument('--param_file', type=str, default="vgg.models",
                       help='do not need specify if need init')
    parse.add_argument('--archid', type=int, default=1, help='the No. of architecture of net, default=1')
    parse.add_argument('--alterconv', type=bool, default=True, help='whether to switch to my define covolution')
    parse.add_argument('--isFLF', type=bool, default=True, help='compress kernel to fixed length in each filter')
    parse.add_argument('--lr', type=float, default=0.0001)

    args = parse.parse_args()
    global sw
    file = os.path.join(os.getcwd(), args.root)
    gls.set_sw(file)
    gls.set_param_for_vgg(args.archid, args.alterconv)
    print 'root file', file
    if not os.path.exists(file):
        os.makedirs(file)
    # sw = SummaryWriter(logdir=file, flush_secs=20)
    print '--arags.FLF:', args.isFLF, '--gpu', args.gpuid, '--lr=', args.lr, \
        '--root=', file, '--param_file=', args.param_file, '--Isinit=', args.Isinit, \
        '--alterconv=', args.alterconv, '--args.archid=', args.archid
    sw = gls.set_sw(file, flush_secs=20)
    train_vgg(gpu=args.gpuid, lr=args.lr, root=file,
              param_file=args.param_file, Isinit=args.Isinit, FLF=args.isFLF)
    '''
    1. origin vgg 
    python train_vggs.py --gpuid=0 --root=vggs_1 --Isinit=True --archid=0  
    2. bigger kernel vgg 
    python train_vggs.py --gpuid=0 --root=vggs_1 --Isinit=True --archid=1  
    #3. bigger kernel vgg; FLF; 
    python train_vggs.py --gpuid=0 --root=vggs_1 --Isinit=True --archid=1  --isFLF=True
    4. bigger kernel vgg; FLF; dynamic pruning 
    python train_vggs.py --gpuid=0 --root=vggs_1 --Isinit=True --archid=1 --alterconv=True --isFLF=True
    '''
