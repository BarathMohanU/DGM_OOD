"""Utilities for Tensorboard logging, checkpointing and Log Likelihood computation."""

import collections
import io
import os
import tempfile
import time

from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
import tqdm

from dataset_utils import get_dataset
gfile = tf.io.gfile
# from google3.third_party.tensorflow.python.ops import summary_ops_v2  # pylint: disable=g-direct-tensorflow-import


class CNSModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  """Wrapper for callback class to save checkpoints to CNS."""

  def _save_model(self, epoch, logs):
    """Saves the model.

    Arguments:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
    logs = logs or {}

    if isinstance(self.save_freq,
                  int) or self.epochs_since_last_save >= self.period:
      self.epochs_since_last_save = 0
      filepath = self._get_file_path(epoch, None, logs)

      if self.save_best_only:
        current = logs.get(self.monitor)
        if current is None:
          logging.warning(
              'Can save best model only with %s available, skipping.',
              self.monitor)
        else:
          if self.monitor_op(current, self.best):
            if self.verbose > 0:
              logging.info(
                  '\nEpoch %d: %s improved from %f to %f, saving model to %s',
                  epoch+1, self.monitor, self.best, current, filepath)

            self.best = current
            if self.save_weights_only:
              save_model_to_cns(self.model, filepath, include_optimizer=False)
            else:
              save_model_to_cns(self.model, filepath)
          else:
            if self.verbose > 0:
              logging.info(
                  '\nEpoch %d: %s did not improve from %f',
                  epoch+1, self.monitor, self.best)
      else:
        if self.verbose > 0:
          logging.info('\nEpoch %d: saving model to %s', epoch+1, filepath)
        if self.save_weights_only:
          save_model_to_cns(self.model, filepath, include_optimizer=False)
        else:
          save_model_to_cns(self.model, filepath)

      self._maybe_remove_file()


# class TensorBoardWithLLStats(tf.keras.callbacks.TensorBoard):
#   """Logs Log Likelihood statistics in the form of histograms and mean AUROC."""

#   def __init__(self, eval_every, id_data, datasets, mode, normalize,
#                visible_dist, **kwargs):
#     super(TensorBoardWithLLStats, self).__init__(**kwargs)
#     self.eval_every = eval_every
#     self.id_data = id_data
#     self.datasets = datasets
#     self.mode = mode
#     self.normalize = normalize
#     self.visible_dist = visible_dist

#   def write_ll_hist(self, epoch, probs_res):
#     plt.subplot(2, 1, 1)
#     for dataset in self.datasets:
#       sns.distplot(probs_res['orig_probs'][dataset], label=dataset)
#     plt.title('Log Likelihood')
#     plt.legend()
#     plt.subplot(2, 1, 2)
#     for dataset in self.datasets:
#       sns.distplot(probs_res['corr_probs'][dataset], label=dataset)
#     plt.title('Corrected Log Likelihood')
#     plt.legend()

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     hist_img = tf.image.decode_png(buf.getvalue(), channels=4)
#     hist_img = tf.expand_dims(hist_img, 0)

#     with summary_ops_v2.always_record_summaries():
#       with self._val_writer.as_default():
#         summary_ops_v2.image('ll_hist', hist_img, step=epoch)

#   def write_auc(self, epoch, probs_res):
#     for probs in probs_res:
#       score_sum = 0
#       for dataset in self.datasets:
#         if dataset == self.id_data:
#           continue
#         targets = np.concatenate([np.zeros_like(probs_res[probs][dataset]),
#                                   np.ones_like(probs_res[probs][self.id_data])])
#         lls = np.concatenate([probs_res[probs][dataset],
#                               probs_res[probs][self.id_data]])
#         score_sum += sklearn.metrics.roc_auc_score(targets, lls)
#       with summary_ops_v2.always_record_summaries():
#         with self._val_writer.as_default():
#           summary_ops_v2.scalar(f'{probs}_auroc_mean',
#                                 score_sum/(len(self.datasets)-1),
#                                 step=epoch)

#   def on_epoch_end(self, epoch, logs=None):
#     logging.info('Epoch %d completed', epoch)
#     if epoch != 0 and epoch % self.eval_every == 0:
#       probs_res = get_probs(self.datasets,
#                             self.model,
#                             self.mode,
#                             self.normalize,
#                             n_samples=10,
#                             split='val',
#                             visible_dist=self.visible_dist)
#       self.write_ll_hist(epoch, probs_res)
#       self.write_auc(epoch, probs_res)
#     super(TensorBoardWithLLStats, self).on_epoch_end(epoch, logs)


def save_model(model, filepath, overwrite=False):
  """Saves a model, buffering in a local file.

  The model is saved on the local or remote Borg scratch disk. The file
  is then copied to the path specified by the user.

  Args:
    model: Keras model instance to be saved.
    filepath: The path to which the model is copied. The directory must already
      exist.
    overwrite: passed to gfile.Copy.

  Raises:
      ImportError: if h5py is not available.
  """

  local_filename = filepath.split('/')[-1]
  parent = tempfile.gettempdir()
  tmpfileprefix = str(int(time.time())) + '_' + local_filename
  tmp_filename = os.path.join(parent, tmpfileprefix)
  print(tmp_filename)
  model.save_weights(tmp_filename)
  if os.path.exists(tmp_filename):
    gfile.copy(tmp_filename, filepath, overwrite=overwrite)
    gfile.remove(tmp_filename)
  else:
    for file in os.listdir(parent):
      if file.startswith(tmpfileprefix):
        gfile.copy(
            os.path.join(parent, file),
            os.path.join(os.path.abspath(os.path.join(filepath, os.pardir)),
                         local_filename+file[len(tmpfileprefix):]),
            overwrite=overwrite)
        gfile.remove(os.path.join(parent, file))


def save_model_to_cns(model, filepath, overwrite=True, include_optimizer=True):
  # Check if the user's filepath is a valid
  # if gfile.pywrapfile.FileName(filepath).is_local():
  #   raise ValueError('You must provide a valid GoogleFile path to save model '
  #                    'checkpoints.')
  include_optimizer = include_optimizer and model.optimizer
  save_model(model, filepath, overwrite=overwrite)


def cb_neglogprob(x, target):
  if x != 0.5:
    c = 2*np.arctanh(1-2*x)/(1-2*x)
  else:
    c = 2
  return -np.log(c * (x**target) * ((1-x)**(1-target)))


def get_probs(train, datasets,
              model,
              mode,
              normalize,
              n_samples,
              split='val',
              training=False,
              visible_dist='cont_bernoulli'):
  """Returns the log likelihoods of examples from given datasets using a given VAE.

  Args:
    id_data: In-distribution dataset name
    datasets: A list of OOD dataset names
    model: A Keras VAE model
    mode: Load in "grayscale" or "color" modes
    normalize: Type of normalization to apply. Supported values are:
      None
      pctile-x (x is an integer)
      histeq
    n_samples: No. of samples to use to compute the importance weighted log
      likelihood estimate
    split: Data split to use for OOD log likelihood estimates,
    training: Get LL estimates using the model in training or eval mode
    visible_dist: VAE's Visible distribution
  Returns:
    A dictionay in the format:
    {
      'orig_probs': {
                      <dataset_name1>: <list of log likelihoods>,
                      <dataset_name2>: <list of log likelihoods>,
                      ...
                    }
      'corr_probs': {
                      <dataset_name1>: <list of log likelihoods>,
                      <dataset_name2>: <list of log likelihoods>,
                      ...
                    }
    }
  """
  logging.info('Computing Log Probs')
  orig_probs = collections.defaultdict(list)
  corr_probs = collections.defaultdict(list)

  for dataset in datasets:
    logging.info('Dataset: %s', dataset)
    if split == 'test':
      _, _, test = get_dataset(dataset, 128, mode=mode,
                               normalize=normalize, dequantize=False,
                               visible_dist=visible_dist)
    elif split == 'val':
      _, test, _ = get_dataset(dataset, 128, mode=mode,
                               normalize=normalize, dequantize=False,
                               visible_dist=visible_dist)
    else:
      test, _, _ = get_dataset(dataset, 128, mode=mode,
                               normalize=normalize, dequantize=False,
                               visible_dist=visible_dist)

    for test_batch in tqdm.tqdm(test):
      inp = test_batch[0]
      target = test_batch[1]

      probs = model.log_prob(
          inp,
          target,
          n_samples=n_samples,
          training=training).numpy()
      orig_probs[dataset].append(probs)
      target = target.numpy()

      if visible_dist == 'categorical':
        target = target.astype(np.int32)
        if model.inp_shape[-1] == 3:
          target[:, :, :, 1:] += 256
          target[:, :, :, 2:] += 256
      if visible_dist in ['gaussian', 'vanilla_gaussian']:
        target[:, :, :, 1:] += 1
        target[:, :, :, 2:] += 1
      corr_probs[dataset].append(probs -
                                 model.correct(target).sum(axis=(1, 2, 3)))
    orig_probs[dataset] = np.concatenate(orig_probs[dataset], axis=0)
    corr_probs[dataset] = np.concatenate(corr_probs[dataset], axis=0)
    
    # I added
    
    test, _, _ = get_dataset(dataset, 128, mode=mode,
                               normalize=normalize, dequantize=False,
                               visible_dist=visible_dist)
    
    for test_batch in tqdm.tqdm(test):
      inp = test_batch[0]
      target = test_batch[1]

      probs = model.log_prob(
          inp,
          target,
          n_samples=n_samples,
          training=training).numpy()
      orig_probs[dataset + '_train'].append(probs)
      target = target.numpy()

      if visible_dist == 'categorical':
        target = target.astype(np.int32)
        if model.inp_shape[-1] == 3:
          target[:, :, :, 1:] += 256
          target[:, :, :, 2:] += 256
      if visible_dist in ['gaussian', 'vanilla_gaussian']:
        target[:, :, :, 1:] += 1
        target[:, :, :, 2:] += 1
      corr_probs[dataset + '_train'].append(probs -
                                 model.correct(target).sum(axis=(1, 2, 3)))
    orig_probs[dataset + '_train'] = np.concatenate(orig_probs[dataset], axis=0)
    corr_probs[dataset + '_train'] = np.concatenate(corr_probs[dataset], axis=0)
    
  return {'orig_probs': orig_probs, 'corr_probs': corr_probs}


@tf.function
def get_pix_ll(batch, model):
  posterior = model.encoder(batch[0], training=False)
  code = posterior.mean()
  visible_dist = model.decoder(code, training=False)
  pix_ll = visible_dist.log_prob(batch[1])
  return pix_ll
