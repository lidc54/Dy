from mxnet.gluon.data import dataset
from mxnet.gluon.data.vision import transforms
from mxnet import image, nd
import matplotlib.pyplot as plt
import numpy as np
import random, os

img_width, img_height = 256, 256
crop_width, crop_height = 224, 224


class crops(transforms.CenterCrop):
    def __init__(self, size, interpolation=1):
        super(crops, self).__init__(size, interpolation)

    def forward(self, x):
        return image.random_crop(x, *self._args)[0]


class loaders(dataset.Dataset):
    def __init__(self, list_path, data_path, mode="train"):
        super(loaders, self).__init__()
        self.list_path = list_path
        # self.read_list(list_path)
        self.path = data_path
        self.transform = [crops(size=(crop_width, crop_height)),
                          transforms.RandomFlipLeftRight(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=0.0, std=0.25)
                          ]
        self.mode = mode

    def __getitem__(self, idx):
        try:
            img_path, label = self.read_list(idx)
            img = self.read_img(img_path)
            return img, int(label)
        except Exception, e:
            print 'error: ', e
            idx = random.randint(0, self.__len__())
            return self.__getitem__(idx)

        # return:img, class weight, attribute values
        # gt_labels, transf_matrix = self.read_label(label, attr)

        # if self.mode != 'train':
        # self.read_json(img_path)
        # try:
        #    keypoint = self.build_canvas()
        #    # self.show_img(img, keypoint)
        #    return img, [self.cls_index, gt_labels], transf_matrix, nd.array(keypoint)
        # except Exception, e:
        #    print('*' * 10, idx, '  loader errore ', e)
        #    return

    def __len__(self):
        with open(self.list_path)as f:
            return len(f.readlines())

    def read_list(self, idx):
        with open(self.list_path)as f:
            line = f.readlines()[idx]
            return line.split()

    def read_img(self, img_path):
        img_path = self.path + img_path
        img = image.imread(img_path)
        self.img_size = img.shape
        img = image.imresize(img, img_width, img_height)

        for trans in self.transform:
            img = trans(img)
        return img


def showImg():
    list_path = "/home1/ImageNet_ILSVRC2012/train_t3.txt"
    data_path = "/home1/ImageNet_ILSVRC2012/ILSVRC2012_img_train_t3/"

    f_d = loaders(list_path, data_path)
    print 'len:', len(f_d)
    for i in range(3):
        idx = random.randint(0, len(f_d))
        print idx, ' '
        # idx = random.choice([70267, 53789, 26727])
        datas = f_d[idx]
        data = datas[0].asnumpy().transpose(1, 2, 0)
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        print 'data shape', data.shape, 'min', np.min(data), 'max', np.max(data)
        # plt.imshow(data.astype(np.uint8))
        # print datas[1:]
        # plt.show()


# def load_Image():
# """special for Image"""

if __name__ == "__main__":
    showImg()
