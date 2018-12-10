# 1. origin vgg 16, without bn layer
- init:no MX_init
- vggs_1
- python train_vggs.py --gpuid=0 --root=vggs_1 --Isinit=True --archid=0
- 这个文件是直接对vgg查看在imagenet上的分类效果
- 对整个vgg的网络参数进行finetune

# 2. vgg 16, no BN, bigger kernel(first 2 layers)
- softmaxwithloss;
- vggs_2
- order: python train_vggs.py --gpuid=0 --root=vggs_1 --Isinit=True --archid=1
- 精度下降到0.6附近
- 这个是对设定的层，比如前4层，使用大卷积核替代的方法。
- 有人提过“层间蒸馏”的方法，直接看的话，可以考虑这种蒸馏方法的训练。
- 但是，这个文件夹更倾向于直接替换的方式。

# 3. vgg 16, no BN, bigger kernel(first 2 layers), constrin No. 
- softmaxwithloss; loss_kernel
- vggs_3:Nov 1; Nov 5; Nov 7 -- finetune twice(1,5)
- this is bigger kernel in first two convolution: 5*5
- and constrain number in them, kept 5 in this two convo
- vggs_31:Nov 11 -- finetue only onece
- abandon
# 4. vgg 16, no BN, bigger kernel(first 2 layers), constrin No., dynamic pruning
- 精度接近0
- vggs_41
- order: python train_vggs.py --gpuid=0 --root=vggs_1 --Isinit=True --archid=1 --alterconv=True --isFLF=True
- 精度也下降到0.6附近
# vggs
- 增加了函数loss_distribution_gt_threshold，放在dy_conv中
- 作用是构建了内圈和外圈的差值，目的是想让kernel的内核和外圈有效参数的比重相等
- 效果是精度接近0
# 随机消除模式
- 增加了一组随机消除模式的方法，目的让这些组合更分散
