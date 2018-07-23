from sphere_net import SphereNet20, AngleLoss, test_accur
from mxnet import gluon, autograd
import mxnet as mx
import data, pickle, os

from layers.dy_conv import origin_conv, new_conv, constrain_kernal_num
from layers.params import global_param, sw, root, alpha
from units import getParmas, init_sphere, load_gamma, load_gamma_test


def train_model():
    my_fun = origin_conv
    save_global_prams = False
    loaded_model = root + "/spherenet_bn_4dy.model"
    loaded_param = root + "/global.param"
    ctx = mx.gpu(6)
    use_bn = True  # whether Batch normalization will be used in the net
    dropout = True  # whether gamma fliter will be random compressed
    # several paramers need update for different duty |
    # and notice params need to be updated

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    data_fold = "/home1/CASIA-WebFace/aligned_Webface-112X96/"
    batch_size = 192
    mnet = SphereNet20(my_fun=my_fun, use_bn=use_bn)
    lr = 0.0003
    stop_epoch = 300

    # initia the net and return paramers of bn -- gamma
    gammas = init_sphere(mnet, loaded_model, ctx)
    # data iteratier
    train_data_loader, valid_data_loader \
        = data.train_valid_test_loader(data_fold, batch_size=batch_size)
    mnet.feature = False  # get features or classes throught spherenet

    Aloss = AngleLoss()
    # L1_loss = gluon.loss.L1Loss()
    # finetune the net
    params = getParmas(mnet, mode='conv')
    trainer = gluon.Trainer(params, 'adam', {'learning_rate': lr})

    # initia the mask for prune
    # mask_type = 'conv'  # 'conv'or 'bn'
    # so try not set these two file in one file
    if os.path.exists(loaded_param):
        with open(loaded_param)as f:
            sv = pickle.load(f)
        global_param.load_param(sv, ctx=ctx)
    else:
        if isinstance(my_fun(1, 1), new_conv):
            global_param.set_param(params.keys(), ctx=ctx)
        elif isinstance(my_fun(1, 1), origin_conv):
            global_param.set_param(gammas, ctx=ctx)

    epoch_train = len(train_data_loader)
    # train
    for epoch in range(stop_epoch):
        j = epoch * epoch_train
        for i, (batch, label) in enumerate(train_data_loader):
            if epoch > 10: dropout = False
            batch = batch.as_in_context(ctx)
            label = label.as_in_context(ctx)
            global_param.iter = i + j
            with autograd.record():
                out = mnet(batch)
                loss_a = Aloss(out[0], out[1], label)
                if use_bn:
                    loss_nums = load_gamma_test(mnet, i + j)
                else:
                    loss_nums = constrain_kernal_num()
                loss = loss_a + loss_nums * alpha
            loss.backward()
            trainer.step(batch_size)
            value2 = loss_nums.asscalar() * alpha
            value = loss_a.asscalar() / batch_size + value2
            sw.add_scalar(tag='Loss', value=value, global_step=i + j)
            sw.add_scalar(tag='K_Loss', value=value2, global_step=i + j)
            if i % 200 == 0:
                print('iter:%d,loss:%4.3f' % (i + j, value))
            # if isinstance(my_fun(1, 1), new_conv):
            #     # get sparse of the dynamic
            #     s_dict = get_sparse_ratio()
            #     for k, v in s_dict:
            #         sw.add_scalar(tag=k, value=v, global_step=j)

        # validation
        for i, (batch, label) in enumerate(valid_data_loader):
            batch = batch.as_in_context(ctx)
            label = label.as_in_context(ctx)
            out = mnet(batch)
            discroc, controc = test_accur(label, i + j, *out)
            sw.add_scalar(tag='RocDisc', value=discroc.asscalar(), global_step=i + j)
            sw.add_scalar(tag='RocCont', value=controc.asscalar(), global_step=i + j)
        # save model
        mnet.save_params(loaded_model)
        if save_global_prams:  # isinstance(my_fun(1, 1), new_conv):
            # save the params of mask
            with open(loaded_param, 'w')as f:
                pickle.dump(global_param, f)


if __name__ == "__main__":
    train_model()
    print('o')
