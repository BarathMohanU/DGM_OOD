"""This module contains some utility functions for loading data."""
"""bm comment"""

import csv
import functools
import os
import random
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import scipy.stats as stats

# from google3.pyglib import gfile
import zca
partial = functools.partial


def noise_generator(split=b'val', mode=b'grayscale'):
  """Generator function for the noise dataest.

  Args:
    split: Data split to load - "train", "val" or "test"
    mode: Load in "grayscale" or "color" modes
  Yields:
    An noise image
  """
  # Keeping support for train to prevent go/pytype-errors#attribute-error
  # during build, but it's not used for training
  if split == b'train':
    np.random.seed(0)
  if split == b'val':
    np.random.seed(1)
  else:
    np.random.seed(2)
  for _ in range(10000):
    if mode == b'grayscale':
      yield np.random.randint(low=0, high=256, size=(32, 32, 1))
    else:
      yield np.random.randint(low=0, high=256, size=(32, 32, 3))
    
def truncnorm_noise_generator(split=b'val', mode=b'grayscale'):
  """Generator function for the noise dataest.

  Args:
    split: Data split to load - "train", "val" or "test"
    mode: Load in "grayscale" or "color" modes
  Yields:
    An noise image
  """
  # Keeping support for train to prevent go/pytype-errors#attribute-error
  # during build, but it's not used for training
  if split == b'train':
    np.random.seed(0)
  if split == b'val':
    np.random.seed(1)
  else:
    np.random.seed(2)
  for _ in range(10000):
    if mode == b'grayscale':
      yield np.round((stats.truncnorm.rvs(-1, 1, size=(32,32,1))+1)*(255/2))
    else:
      yield np.round((stats.truncnorm.rvs(-1, 1, size=(32,32,3))+1)*(255/2))


def compcars_generator(split=b'train'):
  """Generator function for the CompCars Surveillance dataest.

  Source: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

  Args:
    split: Data split to load - "train", "val" or "test".

  Yields:
    An image
  """

  rootpath = '/home/barath/data/compcars/sv_data'
  random.seed(42)
  # random.seed(43)  # split 2

  if split in [b'train', b'val']:
    split_path = os.path.join(rootpath, 'train_surveillance.txt')
    with open(split_path) as f:
      all_images = f.read().split()
    random.shuffle(all_images)
    if split == b'train':
      all_images = all_images[:-(len(all_images)//10)]
    else:
      all_images = all_images[-(len(all_images)//10):]
    for image_name in all_images:
      yield plt.imread(os.path.join(rootpath, 'image', image_name))

  elif split == b'test':
    split_path = os.path.join(rootpath, 'test_surveillance.txt')
    with open(split_path) as f:
      all_images = f.read().split()
    for image_name in all_images:
      yield plt.imread(os.path.join(rootpath, 'image', image_name))


def gtsrb_generator(split=b'train', cropped=False):
  """Generator function for the GTSRB Dataset.

  Source: https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

  Args:
    split: Data split to load - "train", "val" or "test".
    cropped: Whether to load cropped version of the dataset.

  Yields:
    An image
  """

  rootpath = '/home/barath/data/gtsrb/GTSRB'
  random.seed(42)
  # random.seed(43)  # split 2
  if split in [b'train', b'val']:
    rootpath = os.path.join(rootpath, 'Final_Training', 'Images')
    all_images = []
    # loop over all 42 classes
    for c in range(0, 43):
      # subdirectory for class
      prefix = rootpath + '/' + format(c, '05d') + '/'
      # annotations file
      gt_file = open(prefix + 'GT-'+ format(c, '05d') + '.csv')
      # csv parser for annotations file
      gt_reader = csv.reader(gt_file, delimiter=';')
      next(gt_reader)  # skip header

      for row in gt_reader:
        all_images.append((prefix + row[0],
                           (int(row[3]), int(row[4]), int(row[5]), int(row[6]))
                          ))
      gt_file.close()
    random.shuffle(all_images)
    if split == b'train':
      all_images = all_images[:-(len(all_images)//10)]
    else:
      all_images = all_images[-(len(all_images)//10):]
    for image, bbox in all_images:
      img = plt.imread(image)
      if cropped:
        img = img[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1, :]
      yield img

  elif split == b'test':
    rootpath = os.path.join(rootpath, 'Final_Test', 'Images/')
    gt_file = open(rootpath + '/GT-final_test.test.csv')
    gt_reader = csv.reader(gt_file, delimiter=';')
    next(gt_reader)
    for row in gt_reader:
      img = plt.imread(rootpath + row[0])
      if cropped:
        bbox = (int(row[3]), int(row[4]), int(row[5]), int(row[6]))
        img = img[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1, :]
      yield img
    gt_file.close()


def cifar10_class_generator(split, cls):
  """Generator function of class wise CIFAR10 dataset.

  Args:
    split: Data split to load - "train", "val" or "test".
    cls: The target class to load examples from.

  Yields:
    An image
  """
  (ds_train, ds_val, ds_test) = tfds.load('cifar10',
                                          split=['train[:90%]',
                                                 'train[90%:]',
                                                 'test'
                                                 ],
                                          as_supervised=True)
  # split 2
  # (ds_train, ds_val, ds_test) = tfds.load('cifar10',
  #                                         split=['train[10%:]',
  #                                                'train[:10%]',
  #                                                'test'
  #                                                ],
  #                                         as_supervised=True)

  if split == b'train':
    ds = ds_train
  elif split == b'val':
    ds = ds_val
  else:
    ds = ds_test

  for x, y in ds:
    if y == cls:
      yield x

def celeba_generator(split=b'train'):

    rootpath = '/home/barath/data/celeb_a/images'
    random.seed(42)
    all_images = [f for f in os.listdir(rootpath)]
    
    if split in [b'train', b'val']:
        all_images = all_images[:162770]
        random.shuffle(all_images)
        if split == b'train':
            all_images = all_images[:-(len(all_images)//10)]
        else:
            all_images = all_images[-(len(all_images)//10):]
        for image_name in all_images:
            yield plt.imread(os.path.join(rootpath, image_name))

    elif split == b'test':
        all_images = all_images[162770:]
        for image_name in all_images:
            yield plt.imread(os.path.join(rootpath, image_name))
            
def omniglot_shuffled_generator(split=b'train'):
  """Shuffles omniglot train and test examples to eliminate test distribution shift."""
  ds = tfds.load('omniglot', with_info=False)
  all_images = []
  for x in ds['train']:
    all_images.append(x['image'])
  for x in ds['test']:
    all_images.append(x['image'])
  random.seed(42)
  # random.seed(43) # split 2
  n = len(all_images)
  random.shuffle(all_images)
  if split == b'train':
    all_images = all_images[:int(0.8*n)]
  elif split == b'val':
    all_images = all_images[int(0.8*n):int(0.9*n)]
  else:
    all_images = all_images[int(0.9*n):]

  for img in all_images:
    yield img


def notmnist_generator(split=b'train'):
  """Genrates the NotMNIST dataset."""
  rootpath = 'notMNIST_large'
  random.seed(42)
  # random.seed(43) # split 2

  all_images = []
  for classdir in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
    all_images.extend([
        f'{classdir}/{fname}' for fname in os.listdir(f'{rootpath}/{classdir}')
    ])
  random.shuffle(all_images)
  n = len(all_images)
  n = 100000
  if split == b'train':
    all_images = all_images[:int(n*0.8)]
  elif split == b'val':
    all_images = all_images[int(n*0.8):int(n*0.9)]
  else:
    all_images = all_images[int(n*0.9):n]

  for fname in all_images:
    yield plt.imread(os.path.join(rootpath, fname))[:, :, None] * 255

def sign_lang_generator(split=b'val'):
    
    rootpath = '/home/barath/data/sign_lang/Hand_sign_mnist/'
    random.seed(42)
    
    if split in [b'train', b'val']:
        all_images = []
        for root, dirs, files in os.walk(rootpath + 'Train'):
             for file in files:
                all_images.append(os.path.join(root, file))
        random.shuffle(all_images)
        if split == b'train':
            all_images = all_images[:-(len(all_images)//10)]
        else:
            all_images = all_images[-(len(all_images)//10):]
        for image_name in all_images:
            yield np.expand_dims(plt.imread(os.path.join(image_name)),-1)

    elif split == b'test':
        all_images = []
        for root, dirs, files in os.walk(rootpath + 'Test'):
             for file in files:
                all_images.append(os.path.join(root, file))
        for image_name in all_images:
            yield np.expand_dims(plt.imread(os.path.join(image_name)),-1)

def get_dataset(name,
                batch_size,
                mode,
                normalize=None,
                dequantize=False,
                shuffle_train=True,
                visible_dist='cont_bernoulli', 
                zca_transform=None,
                mutation_rate=None
                ):
  """Returns the required dataset with custom pre-processing.

  Args:
    name: Name of the dataset. Supported names are:
      svhn_cropped
      cifar10
      celeb_a
      gtsrb
      gtsrb_cropped
      compcars
      mnist
      fashion_mnist
      omniglot_inverted
      omniglot_shuffled_inverted
      emnist/letters
      kmnist
      noise
    batch_size: Batch Size
    mode: Load in "grayscale" or "color" modes
    normalize: Type of normalization to apply. Supported values are:
      None
      pctile-x (x is an integer)
      histeq
    dequantize: Whether to apply uniform dequantization
    shuffle_train: Whether to shuffle examples in the train split
    visible_dist: Visible dist of the model

  Returns:
    The train, val and test splits respectively
  """

  def preprocess(image, inverted, mode, normalize, dequantize, visible_dist, zca_transform=None, mutation_rate=None):
    if isinstance(image, dict):
      image = image['image']
    
    if mutation_rate is not None:
        image = tf.image.resize(image, [32, 32], antialias=True)
        image = tf.cast(tf.round(image), tf.int32)
        if mutation_rate != 0:
            w, h, c = image.get_shape().as_list()
            mask = tf.cast(
                  tf.compat.v1.multinomial(
                      tf.compat.v1.log([[1.0 - mutation_rate, mutation_rate]]), w * h * c),
                  tf.int32)[0]
            mask = tf.reshape(mask, [w, h, c])
            possible_mutations = tf.compat.v1.random_uniform(
                  [w * h * c],
                  minval=0,
                  maxval=256,
                  dtype=tf.int32)
            possible_mutations = tf.reshape(possible_mutations, [w, h, c])
            image = tf.compat.v1.mod(image + mask * possible_mutations, 256)
        image = tf.cast(tf.round(image), tf.float32)
        return image
    
    image = tf.cast(image, tf.float32)
    if dequantize:
      image += tf.random.uniform(image.shape)
      image = image / 256.0
    else:
      image = image / 255.0
    image = tf.image.resize(image, [32, 32], antialias=True)
    if mode == 'grayscale':
      if image.shape[-1] != 1:
        image = tf.image.rgb_to_grayscale(image)
    else:
      if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)

    if isinstance(normalize, str) and normalize.startswith('pctile') and not normalize.endswith('channelwhiten'):
      pct = float(normalize.split('-')[1].split('+')[0])
      mn = tfp.stats.percentile(image, pct)
      mx = tfp.stats.percentile(image, 100-pct)
      if mx == mn:
        mn = tfp.stats.percentile(image, 0)
        mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
    elif normalize == 'histeq':
      # pylint: disable=g-long-lambda
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(exposure.equalize_hist(x.numpy()),
                                         dtype=tf.float32),
          [image],
          tf.float32
      )
      # pylint: enable=g-long-lambda
    elif normalize == 'adhisteq':
      # pylint: disable=g-long-lambda
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(
              exposure.equalize_adapthist(x.numpy().squeeze())[:, :, None],
              dtype=tf.float32), [image], tf.float32)
    elif normalize == 'channelwhiten':
      if mode == 'color':
        n_channels = 3
      else:
         n_channels = 1
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(
              zca.ZCA().fit(x.numpy().reshape((1024, n_channels)))
              .transform(x.numpy().reshape(1024, n_channels)).reshape((32,32,n_channels)),
              dtype=tf.float32),
          [image],
          tf.float32
      )
      mn = tfp.stats.percentile(image, 0)
      mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
    elif normalize == 'zca_original':
      if mode == 'color':
        n_channels = 3
      else:
         n_channels = 1
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(
              zca_transform.transform(x.numpy().reshape((1,1024*n_channels))).reshape((32,32,n_channels)),
              dtype=tf.float32),
          [image],
          tf.float32
      )
      mn = tfp.stats.percentile(image, 0)
      mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
        
    # for pairs
    elif normalize == 'channelwhiten+pctile-5':
      if mode == 'color':
        n_channels = 3
      else:
         n_channels = 1
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(
              zca.ZCA().fit(x.numpy().reshape((1024, n_channels)))
              .transform(x.numpy().reshape(1024, n_channels)).reshape((32,32,n_channels)),
              dtype=tf.float32),
          [image],
          tf.float32
      )
      mn = tfp.stats.percentile(image, 0)
      mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
      pct = 5
      mn = tfp.stats.percentile(image, pct)
      mx = tfp.stats.percentile(image, 100-pct)
      if mx == mn:
        mn = tfp.stats.percentile(image, 0)
        mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
    elif normalize == 'pctile-5+channelwhiten':
      pct = 5
      mn = tfp.stats.percentile(image, pct)
      mx = tfp.stats.percentile(image, 100-pct)
      if mx == mn:
        mn = tfp.stats.percentile(image, 0)
        mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
      if mode == 'color':
        n_channels = 3
      else:
         n_channels = 1
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(
              zca.ZCA().fit(x.numpy().reshape((1024, n_channels)))
              .transform(x.numpy().reshape(1024, n_channels)).reshape((32,32,n_channels)),
              dtype=tf.float32),
          [image],
          tf.float32
      )
      mn = tfp.stats.percentile(image, 0)
      mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
    elif normalize == 'channelwhiten+histeq':
      if mode == 'color':
        n_channels = 3
      else:
         n_channels = 1
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(
              zca.ZCA().fit(x.numpy().reshape((1024, n_channels)))
              .transform(x.numpy().reshape(1024, n_channels)).reshape((32,32,n_channels)),
              dtype=tf.float32),
          [image],
          tf.float32
      )
      mn = tfp.stats.percentile(image, 0)
      mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
      # pylint: disable=g-long-lambda
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(exposure.equalize_hist(x.numpy()),
                                         dtype=tf.float32),
          [image],
          tf.float32
      )
      # pylint: disable=g-long-lambda
    elif normalize == 'histeq+channelwhiten':
      # pylint: disable=g-long-lambda
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(exposure.equalize_hist(x.numpy()),
                                         dtype=tf.float32),
          [image],
          tf.float32
      )
      # pylint: disable=g-long-lambda
      if mode == 'color':
        n_channels = 3
      else:
         n_channels = 1
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(
              zca.ZCA().fit(x.numpy().reshape((1024, n_channels)))
              .transform(x.numpy().reshape(1024, n_channels)).reshape((32,32,n_channels)),
              dtype=tf.float32),
          [image],
          tf.float32
      )
      mn = tfp.stats.percentile(image, 0)
      mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
    elif normalize is not None:
      raise NotImplementedError(
          f'Normalization method {normalize} not implemented')
      # pylint: enable=g-long-lambda
    if inverted:
      image = 1 - image
    image = tf.clip_by_value(image, 0., 1.)
    
    target = image
#     if visible_dist == 'categorical':
#       target = tf.cast(target*255, tf.int32)
    if visible_dist in ['categorical', 'logistic', 'mixture_of_logistics']:
      # target = tf.cast(tf.cast(target, tf.int32), tf.float32)
      target = tf.round(target*255)   # fix this for categorical VAE
      if visible_dist in ['logistic', 'mixture_of_logistics']:
        image = target
    return image, target

  assert name in ['svhn_cropped', 'cifar10', 'celeb_a', 'gtsrb', 'compcars',
                  'mnist', 'fashion_mnist', 'omniglot_inverted', 'sign_lang',
                  'omniglot_shuffled_inverted', 'kmnist', 'emnist/letters',
                  'notmnist', 'noise', 'truncnorm_noise', *[f'cifar10-{i}' for i in range(10)]],\
      f'Dataset {name} not supported'

  inverted = False
  if name.endswith('inverted'):
    name = name[:-9]
    inverted = True

  if name == 'noise':
    n_channels = 1 if mode == 'grayscale' else 3
    # Loading ds_train to prevent go/pytype-errors#attribute-error
    # during build, but it's not used for training
    ds_train = tf.data.Dataset.from_generator(
        noise_generator,
        args=['train', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_val = tf.data.Dataset.from_generator(
        noise_generator,
        args=['val', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_test = tf.data.Dataset.from_generator(
        noise_generator,
        args=['test', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    n_examples = 1024
  elif name == 'truncnorm_noise':
    n_channels = 1 if mode == 'grayscale' else 3
    # Loading ds_train to prevent go/pytype-errors#attribute-error
    # during build, but it's not used for training
    ds_train = tf.data.Dataset.from_generator(
        truncnorm_noise_generator,
        args=['train', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_val = tf.data.Dataset.from_generator(
        truncnorm_noise_generator,
        args=['val', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_test = tf.data.Dataset.from_generator(
        truncnorm_noise_generator,
        args=['test', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    n_examples = 1024
  elif name.startswith('gtsrb'):
    ds_train = tf.data.Dataset.from_generator(
        gtsrb_generator,
        args=['train', name.endswith('cropped')],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        gtsrb_generator,
        args=['val', name.endswith('cropped')],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        gtsrb_generator,
        args=['test', name.endswith('cropped')],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name == 'compcars':
    ds_train = tf.data.Dataset.from_generator(
        compcars_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        compcars_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        compcars_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name == 'celeb_a':
    ds_train = tf.data.Dataset.from_generator(
        celeba_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        celeba_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        celeba_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name == 'sign_lang':
    ds_train = tf.data.Dataset.from_generator(
        sign_lang_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    ds_val = tf.data.Dataset.from_generator(
        sign_lang_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    ds_test = tf.data.Dataset.from_generator(
        sign_lang_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    n_examples = 1024
  elif name.startswith('cifar10-'):
    n_examples = 1024
    cls = int(name.split('-')[1])
    ds_train = tf.data.Dataset.from_generator(
        cifar10_class_generator,
        args=['train', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        cifar10_class_generator,
        args=['val', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        cifar10_class_generator,
        args=['test', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
  elif name.startswith('omniglot_shuffled'):
    ds_train = tf.data.Dataset.from_generator(
        omniglot_shuffled_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        omniglot_shuffled_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        omniglot_shuffled_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name == 'notmnist':
    ds_train = tf.data.Dataset.from_generator(
        notmnist_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    ds_val = tf.data.Dataset.from_generator(
        notmnist_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    ds_test = tf.data.Dataset.from_generator(
        notmnist_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 1))
    n_examples = 1024
  else:
    # (ds_train, ds_val, ds_test), ds_info = tfds.load(
    #     name, split=['train[:90%]', 'train[90%:]', 'test'], with_info=True)
    # split 2
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        name, split=['train[10%:]', 'train[:10%]', 'test'], with_info=True)
    n_examples = ds_info.splits['train'].num_examples

  ds_train = ds_train.map(
      partial(preprocess, inverted=inverted, mode=mode, normalize=normalize,
              dequantize=dequantize, visible_dist=visible_dist, zca_transform=zca_transform, mutation_rate=mutation_rate),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.cache()
  if shuffle_train:
    ds_train = ds_train.shuffle(n_examples)
  ds_train = ds_train.batch(batch_size)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_val = ds_val.map(
      partial(preprocess, inverted=inverted, mode=mode, normalize=normalize,
              dequantize=dequantize, visible_dist=visible_dist, zca_transform=zca_transform, mutation_rate=mutation_rate),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_val = ds_val.cache()
  ds_val = ds_val.batch(batch_size)
  ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.map(
      partial(preprocess, inverted=inverted, mode=mode, normalize=normalize,
              dequantize=dequantize, visible_dist=visible_dist, zca_transform=zca_transform, mutation_rate=mutation_rate),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.batch(batch_size)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  return ds_train, ds_val, ds_test
