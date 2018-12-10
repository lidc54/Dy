from gluoncv.data import ImageNet
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms


def work():
    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
    # https://gluon-cv.mxnet.io/build/examples_datasets/imagenet.html
    # You need to specify ``root`` for ImageNet if you extracted the images into
    # a different folder, not use rec but general images, such as  *.jpg
    train_data = DataLoader(
        ImageNet(train=True, root="/home1/ImageNet_ILSVRC2012/ILSVRC2012_img_train/").transform_first(train_trans),
        batch_size=128, shuffle=True)

    for x, y in train_data:
        print(x.shape, y.shape)
        break
    from gluoncv.utils import viz

    val_dataset = ImageNet(train=False)
    viz.plot_image(val_dataset[1234][0])  # index 0 is image, 1 is label
    viz.plot_image(val_dataset[4567][0])


def notwork():
    import mxnet as mx
    import matplotlib.pyplot as plt
    shape = 224
    train_iter = mx.image.ImageDetIter(
        batch_size=4,
        data_shape=(3, shape, shape),
        path_imgrec="/home1/ImageNet_ILSVRC2012/train_label.rec",
        path_imgidx="/home1/ImageNet_ILSVRC2012/train_label.idx",
        shuffle=False
    )
    train_iter.reset()

    batch = train_iter.next()

    img, labels = batch.data[0], batch.label[0]

    print(labels.shape)

    img = img.transpose((0, 2, 3, 1))
    img = img.clip(0, 255).asnumpy() / 255

    for i in range(4):
        # _, fig = plt.subplots()
        plt.imshow(img[i])
        plt.show()


if __name__ == "__main__":
    notwork()
