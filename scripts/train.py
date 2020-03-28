#!/usr/bin/env python

# mainly copied from chainer/train_imagenet.py
# https://github.com/chainer/chainer/blob/master/examples/imagenet/train_imagenet.py

import argparse

import chainer
from chainer import dataset
from chainer import training
from chainer.links import VGG16Layers
import chainer.backends.cuda
from chainer.serializers import npz
from chainer.training import extensions

from vgg16.vgg16_batch_normalization import VGG16BatchNormalization as VGG16

import matplotlib
import numpy as np
import os.path as osp
from PIL import Image as Image_
import rospkg

matplotlib.use('Agg')  # necessary not to raise Tcl_AsyncDelete Error


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype(chainer.get_dtype())
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]  # (3, 256, 256), rgb
        image -= self.mean
        image *= (1.0 / 255.0)  # Scale to [0, 1.0]
        return image, label


def main():
    rospack = rospkg.RosPack()

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')

    args = parser.parse_args()

    # Configs for train with chainer
    device = chainer.cuda.get_device_from_id(args.gpu)  # for python2
    batchsize = 32
    # Root directory path of image dataset
    root = osp.join(rospack.get_path(
        'decopin_hand'), 'train_data', 'dataset')
    # Output directory of train result
    out = osp.join(rospack.get_path('decopin_hand'),
                   'scripts', 'result')
    # Path to mean image of dataset
    mean_img_path = osp.join(rospack.get_path('decopin_hand'),
                             'train_data', 'dataset', 'mean_of_dataset.png')
    # Path to training image-label list file
    train_labels = osp.join(rospack.get_path('decopin_hand'),
                            'train_data', 'dataset', 'train_images.txt')
    # Path to validation image-label list file
    val_labels = osp.join(rospack.get_path('decopin_hand'),
                          'train_data', 'dataset', 'test_images.txt')

    # Initialize the model to train
    print('Device: {}'.format(device))
    print('Dtype: {}'.format(chainer.config.dtype))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    n_class = 0
    with open(osp.join(root, 'n_class.txt'), mode='r') as f:
        for row in f:
            n_class += 1
    model = VGG16(n_class=n_class)
    model_path = osp.join(rospack.get_path('decopin_hand'), 'scripts', 'VGG_ILSVRC_16_layers.npz')
    if not osp.exists(model_path):
        from chainer.dataset import download
        from chainer.links.caffe.caffe_function import CaffeFunction
        path_caffemodel = download.cached_download('http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel')
        caffemodel = CaffeFunction(path_caffemodel)
        npz.save_npz(model_path, caffemodel, compression=False)

    # TODO: add disable_update to original vgg16 layers
    vgg16 = VGG16Layers(pretrained_model=model_path)
    print('Load model from {}'.format(model_path))
    for l in model.children():
        if l.name.startswith('conv'):
            l1 = getattr(vgg16, l.name)
            l2 = getattr(model, l.name)
            assert l1.W.shape == l2.W.shape
            assert l1.b.shape == l2.b.shape
            l2.W.data[...] = l1.W.data[...]
            l2.b.data[...] = l1.b.data[...]
        elif l.name in ['fc6', 'fc7']:
            l1 = getattr(vgg16, l.name)
            l2 = getattr(model, l.name)
            assert l1.W.size == l2.W.size
            assert l1.b.size == l2.b.size
            l2.W.data[...] = l1.W.data.reshape(l2.W.shape)[...]
            l2.b.data[...] = l1.b.data.reshape(l2.b.shape)[...]

    model.to_device(device)
    device.use()

    # Load mean value of dataset
    mean = np.array(Image_.open(mean_img_path), np.float32).transpose(
        (2, 0, 1))  # (height, width, channel) -> (channel ,height, width), rgb

    # Load the dataset files
    train = PreprocessedDataset(train_labels, root, mean)
    val = PreprocessedDataset(val_labels, root, mean, False)
    # These iterators load the images with subprocesses running in parallel
    # to the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, batchsize)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, batchsize, repeat=False)
    converter = dataset.concat_examples

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out)

    val_interval = 100, 'iteration'
    log_interval = 100, 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, converter=converter,
                                        device=device), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        target=model, filename='model_best.npz'),
        trigger=chainer.training.triggers.MinValueTrigger(
            key='validation/main/loss',
            trigger=val_interval))
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='iteration', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='iteration', file_name='accuracy.png'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    main()
