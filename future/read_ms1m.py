import mxnet as mx
from mxnet import nd
import matplotlib.pyplot as plt
import numpy as np
from mxnet.gluon.data.vision import datasets, transforms


# https://mxnet.incubator.apache.org/api/scala/io.html

# class adsize(transforms.Resize):
#     def __init__(self, size, keep_ratio=False, interpolation=1):
#         super(adsize, self).__init__(size, keep_ratio, interpolation)
#
#     def forward(self, x):
#         x =
#         return nd.transpose(x, axes=(2, 0, 1))


class trans(transforms.Normalize):
    def __init__(self, mean, std):
        super(trans, self).__init__(mean, std)

    def hybrid_forward(self, F, x):
        x = mx.symbol.transpose(x, axes=(2, 0, 1))
        return super(trans, self).hybrid_forward(F, x)


train_transforms = transforms.Compose([#transforms.Resize((96, 112)),
                                       transforms.RandomFlipLeftRight(),
                                       # transforms.ToTensor(),
                                       # adsize(size=(96, 112)),  # W x H
                                       # transforms.Resize((96, 112)),
                                       # trans(mean=0.5, std=0.25),
                                       ])
data_iter = mx.image.ImageIter(batch_size=4, data_shape=(3, 112, 112), shuffle=True,
                               path_imgrec="/home1/face_identify/faces_ms1m_112x112/train.rec",
                               path_imgidx="/home1/face_identify/faces_ms1m_112x112/train.idx",
                               aug_list=train_transforms)
try:
    print 'len1:', len(data_iter.imgidx)
except Exception, e:
    print e
# try:
# print 'len2: ', len(data_iter.imgrec)
# except Exception,e:
# print e
# data_iter:mxnet.image.ImageIter
# reset() resents the iterator to the beginning of the data
data_iter.reset()

# batch type is mxnet.io.DataBatch next() DataBatch
batch = data_iter.next()

data = (batch.data[0]).asnumpy().astype(np.uint8)
# batch.label
# data=train_transforms(data)
# build rec img2rec
# https://mxnet.incubator.apache.org/faq/recordio.html?highlight=im2rec
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(data[i].transpose((1, 2, 0)))
plt.show()
