"""This module implements model classes for VAE and PixelCNN++ models."""

import collections

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.special
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
import tqdm

import utils

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class VAE(tfk.Model):
  """Generic Variational Autoencoder."""

  def __init__(self,
               input_shape,
               latent_dim,
               visible_dist,
               encoder,
               decoder):
    super(VAE, self).__init__()

    self.latent_dim = latent_dim
    self.inp_shape = input_shape
    self.visible_dist = visible_dist

    self.latent_prior = tfd.MultivariateNormalDiag(
        loc=tf.zeros(self.latent_dim),
        scale_diag=tf.ones(self.latent_dim)
    )

    self.encoder = encoder
    self.decoder = decoder

  def call(self, inputs, training=False):
    self.posterior = self.encoder(inputs, training=training)
    self.code = self.posterior.sample()
    self.decoder_likelihood = self.decoder(self.code, training=training)
    return {'posterior': self.posterior, 'decoder_ll': self.decoder_likelihood}

  def compute_corrections(self, dataset=None):
    # pylint: disable=g-long-lambda
    if self.visible_dist == 'cont_bernoulli':
      self.corr_dict = {}
      targets = np.round(np.linspace(1e-3, 1-1e-3, 999), decimals=3)
      for target in targets:
        self.corr_dict[target] = -utils.cb_neglogprob(
            scipy.optimize.fmin(utils.cb_neglogprob, 0.5, args=(target,),
                                disp=False)[0],
            target)
      corr_func = lambda pix: self.corr_dict[(np.clip(pix, 1e-3, 1-1e-3)
                                              .astype(float)
                                              .round(decimals=3)
                                             )].astype(np.float32)
      self.correct = np.vectorize(corr_func)
    elif self.visible_dist == 'bernoulli':
      self.corr_dict = dict(zip(
          np.round(np.linspace(1e-3, 1-1e-3, 999), decimals=3),
          tfd.Bernoulli(probs=tf.linspace(1e-3, 1-1e-3, 999)).prob(
              tf.linspace(1e-3, 1-1e-3, 999)).numpy()
          ))
      corr_func = lambda pix: self.corr_dict[(np.clip(pix, 1e-3, 1-1e-3)
                                              .astype(float)
                                              .round(decimals=3)
                                             )].astype(np.float32)
      self.correct = np.vectorize(corr_func)
    elif self.visible_dist in ['gaussian', 'vanilla_gaussian']:
      assert dataset is not None, ('dataset is required to compute correction '
                                   'for Gaussian visible distribution.')
      self.corr_dict = collections.defaultdict(list)
      update_dict = lambda corr_dict_img: [
          self.corr_dict[pix].append(
              scipy.special.logsumexp(corr_dict_img[pix]) - np.log(
                  len(corr_dict_img[pix]))) for pix in corr_dict_img
      ]
      j = 0
      for train_batch in tqdm.tqdm(dataset):
        j += 1
        pixel_ll = utils.get_pix_ll(train_batch, self)
        inp = train_batch[1].numpy()
        if self.inp_shape[-1] == 3:
          inp[:, :, :, 1:] += 1
          inp[:, :, :, 2:] += 1
        for i in range(inp.shape[0]):
          corr_dict_img = collections.defaultdict(list)
          np.vectorize(
              lambda pix, ll: corr_dict_img[np.round(pix, decimals=2)].append(ll
                                                                             ),
              otypes=[float])(inp[i], pixel_ll[i])
          update_dict(corr_dict_img)

        if j == 500:
          break

      for key in self.corr_dict:
        n = len(self.corr_dict[key])
        self.corr_dict[key] = scipy.special.logsumexp(
            np.array(self.corr_dict[key])) - np.log(n)

      f = scipy.interpolate.interp1d(
          *list(zip(*[(pix, corr) for pix, corr in self.corr_dict.items()])),
          fill_value='extrapolate')
      for missing_pix in (set(np.round(np.linspace(0, 3, 301), 2)) -
                          set(self.corr_dict)):
        self.corr_dict[missing_pix] = f(missing_pix)
      self.correct = np.vectorize(lambda x: self.corr_dict[np.round(x, 2)])
    elif self.visible_dist == 'categorical':
      # n_samples = 100
      assert dataset is not None, ('dataset is required to compute correction '
                                   'for Categorical visible distribution.')
      self.corr_dict = collections.defaultdict(list)
      update_dict = lambda corr_dict_img: [
          self.corr_dict[pix].append(
              scipy.special.logsumexp(corr_dict_img[pix])-np.log(
                  len(corr_dict_img[pix]))) for pix in corr_dict_img
      ]
      # pylint: enable=g-long-lambda
      j = 0
      for train_batch in tqdm.tqdm(dataset):
        j += 1
        pixel_ll = utils.get_pix_ll(train_batch, self)
        inp = train_batch[1].numpy()
        if inp.max() <= 1:
          inp = (inp*255).astype(np.int32)
        if self.inp_shape[-1] == 3:
          inp[:, :, :, 1:] += 256
          inp[:, :, :, 2:] += 256
        for i in range(inp.shape[0]):
          corr_dict_img = collections.defaultdict(list)
          np.vectorize(lambda pix, ll: corr_dict_img[int(pix)].append(ll),
                       otypes=[float])(inp[i], pixel_ll[i])
          update_dict(corr_dict_img)
        if j == 500:
          break

      for key in self.corr_dict:
        n = len(self.corr_dict[key])
        self.corr_dict[key] = scipy.special.logsumexp(
            np.array(self.corr_dict[key])) - np.log(n)
      f = scipy.interpolate.interp1d(
          *list(zip(*[(pix, corr) for pix, corr in self.corr_dict.items()])),
          fill_value='extrapolate')
      for missing_pix in (set(range(256 * self.inp_shape[-1])) -
                          set(self.corr_dict)):
        self.corr_dict[missing_pix] = f(missing_pix)
      self.correct = np.vectorize(lambda x: self.corr_dict[x])

  def kl_divergence_loss(self, target, posterior):
    kld = tfd.kl_divergence(posterior, self.latent_prior)
    return tf.reduce_mean(kld)

  def decoder_nll_loss(self, target, decoder_likelihood):
    decoder_nll = -(decoder_likelihood.log_prob(target))
    decoder_nll = tf.reduce_sum(decoder_nll, axis=[1, 2, 3])
    return tf.reduce_mean(decoder_nll)

  def log_prob(self, inp, target, n_samples, training=False,
               only_decoder_ll=False):
    """Computes an importance weighted log likelihood estimate."""
    posterior = self.encoder(inp, training=training)
    code = posterior.sample(n_samples)
    kld = posterior.log_prob(code) - self.latent_prior.log_prob(code)
    visible_dist = self.decoder(
        tf.reshape(code, [-1, self.latent_dim]), training=training)
    target_rep = tf.reshape(
        tf.repeat(tf.expand_dims(target, 0), n_samples, 0),
        [-1] + list(self.inp_shape))
    decoder_ll = visible_dist.log_prob(target_rep)

    decoder_ll = tf.reshape(decoder_ll, [n_samples, -1] + list(self.inp_shape))
    if only_decoder_ll:  # remove this?
      return tf.reduce_logsumexp(decoder_ll, axis=[0])
    decoder_ll = tf.reduce_sum(decoder_ll, axis=[2, 3, 4])

    elbo = tf.reduce_logsumexp(decoder_ll - kld, axis=0)
    elbo = elbo - tf.math.log(tf.cast(n_samples, dtype=tf.float32))
    return elbo


class CVAE(VAE):
  """Convolutional Variational Autoencoder."""

  def __init__(self, input_shape, num_filters, latent_dim, visible_dist):
    num_channels = input_shape[-1]
    encoder = tfk.Sequential(
        [
            tfkl.InputLayer(input_shape=input_shape),
            tfkl.Conv2D(filters=num_filters, kernel_size=4, strides=2,
                        padding='SAME'),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(filters=2*num_filters, kernel_size=4, strides=2,
                        padding='SAME'),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(filters=4*num_filters, kernel_size=4, strides=2,
                        padding='SAME'),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(filters=2*latent_dim, kernel_size=4, strides=1,
                        padding='VALID'),
            tfkl.Flatten(),
            # pylint: disable=g-long-lambda
            tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
                loc=t[..., :latent_dim],
                scale_diag=tf.nn.softplus(t[..., latent_dim:])
                ))
            # pylint: enable=g-long-lambda
        ]
    )

    decoder_head = []
    if visible_dist == 'cont_bernoulli':
      decoder_head.append(
          tfkl.Conv2DTranspose(filters=num_channels, kernel_size=4,
                               strides=2, padding='SAME')
          )
      decoder_head.append(
          # pylint: disable=g-long-lambda
          tfpl.DistributionLambda(lambda t: tfd.ContinuousBernoulli(
              logits=tf.clip_by_value(t, -15.94, 15.94), validate_args=True
              ), convert_to_tensor_fn=lambda s: s.logits)
          # pylint: enable=g-long-lambda
          )
    if visible_dist == 'bernoulli':
      decoder_head.append(
          tfkl.Conv2DTranspose(filters=num_channels, kernel_size=4,
                               strides=2, padding='SAME')
          )
      decoder_head.append(
          # pylint: disable=g-long-lambda
          tfpl.DistributionLambda(lambda t: tfd.Bernoulli(
              logits=tf.clip_by_value(t, -15.94, 15.94), validate_args=False
              ), convert_to_tensor_fn=lambda s: s.logits)
          # pylint: enable=g-long-lambda
          )
    elif visible_dist == 'gaussian':
      decoder_head.append(
          tfkl.Conv2DTranspose(filters=num_channels, kernel_size=4,
                               strides=2, padding='SAME', activation='sigmoid')
          )
      decoder_head.append(
          # pylint: disable=g-long-lambda
          tfpl.DistributionLambda(lambda t: tfd.TruncatedNormal(
              loc=t, scale=0.2, low=0, high=1,
              ), convert_to_tensor_fn=lambda s: s.loc)
          # pylint: enable=g-long-lambda
          )
    elif visible_dist == 'vanilla_gaussian':
      decoder_head.append(
          tfkl.Conv2DTranspose(filters=num_channels, kernel_size=4,
                               strides=2, padding='SAME', activation='sigmoid')
          )
      decoder_head.append(
          # pylint: disable=g-long-lambda
          tfpl.DistributionLambda(lambda t: tfd.Normal(
              loc=t, scale=0.2,
              ), convert_to_tensor_fn=lambda s: s.loc)
          # pylint: enable=g-long-lambda
          )
    elif visible_dist == 'categorical':
      decoder_head.append(
          tfkl.Conv2DTranspose(filters=num_channels*256, kernel_size=4,
                               strides=2, padding='SAME')
          )
      decoder_head.append(tfkl.Reshape(list(input_shape) + [256]))
      decoder_head.append(
          # pylint: disable=g-long-lambda
          tfpl.DistributionLambda(lambda t: tfd.Categorical(
              logits=t, validate_args=True
              ), convert_to_tensor_fn=lambda s: s.logits)
          # pylint: enable=g-long-lambda
          )

    decoder = tfk.Sequential(
        [
            tfkl.InputLayer(input_shape=(latent_dim,)),
            tfkl.Reshape([1, 1, latent_dim]),
            tfkl.Conv2DTranspose(filters=4*num_filters, kernel_size=4,
                                 strides=1, padding='VALID'),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(filters=2*num_filters, kernel_size=4,
                                 strides=2, padding='SAME'),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(filters=num_filters, kernel_size=4,
                                 strides=2, padding='SAME'),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            *decoder_head
        ]
    )

    super(CVAE, self).__init__(input_shape,
                               latent_dim,
                               visible_dist,
                               encoder,
                               decoder)


class MLPVAE(VAE):
  """MLP Variational Autoencoder."""

  def __init__(self, input_shape, hidden_dim, latent_dim, visible_dist):
    assert visible_dist == 'cont_bernoulli'
    encoder = tfk.Sequential(
        [
            tfkl.InputLayer(input_shape=input_shape),
            tfkl.Flatten(),
            tfkl.Dense(units=hidden_dim),
            tfkl.ReLU(),
            tfkl.Dense(units=2*latent_dim),
            # pylint: disable=g-long-lambda
            tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
                loc=t[..., :latent_dim],
                scale_diag=tf.nn.softplus(t[..., latent_dim:])
                ))
            # pylint: enable=g-long-lambda
        ]
    )

    decoder = tfk.Sequential(
        [
            tfkl.InputLayer(input_shape=(latent_dim,)),
            tfkl.Dense(units=hidden_dim),
            tfkl.ReLU(),
            tfkl.Dense(units=input_shape[0]*input_shape[1]*input_shape[2]),
            tfkl.Reshape(input_shape),
            # pylint: disable=g-long-lambda
            tfpl.DistributionLambda(lambda t: tfd.ContinuousBernoulli(
                logits=tf.clip_by_value(t, -15.94, 15.94), validate_args=True
                ), convert_to_tensor_fn=lambda s: s.logits)
            # pylint: enable=g-long-lambda
        ]
    )

    super(MLPVAE, self).__init__(input_shape,
                                 latent_dim,
                                 visible_dist,
                                 encoder,
                                 decoder)


class PixelCNNwithCB(tfd.PixelCNN):
  """Modidied PixelCNN++ distribution with continuous Bernoulli visible layer."""

  def __init__(self, *args, **kwargs):
    kwargs['num_logistic_mix'] = 1
    kwargs['high'] = 1
    kwargs['low'] = 0
    super(PixelCNNwithCB, self).__init__(*args, **kwargs)

  def _make_mixture_dist(self, component_logits, locs, scales):
    dist = tfd.ContinuousBernoulli(
        logits=tf.clip_by_value(locs[:, :, :, 0, :], -15.94, 15.94))
    return tfd.Independent(dist, reinterpreted_batch_ndims=3)

  def _sample_n(self, n, seed=None, conditional_input=None, training=False):
    if conditional_input is not None:
      conditional_input = tf.convert_to_tensor(
          conditional_input, dtype=self.dtype)
      conditional_event_rank = tensorshape_util.rank(self.conditional_shape)
      conditional_input_shape = prefer_static.shape(conditional_input)
      conditional_sample_rank = prefer_static.rank(
          conditional_input) - conditional_event_rank

      # If `conditional_input` has no sample dimensions, prepend a sample
      # dimension
      if conditional_sample_rank == 0:
        conditional_input = conditional_input[tf.newaxis, ...]
        conditional_sample_rank = 1

      # Assert that the conditional event shape in the `PixelCnnNetwork` is the
      # same as that implied by `conditional_input`.
      conditional_event_shape = conditional_input_shape[
          conditional_sample_rank:]
      with tf.control_dependencies([
          tf.assert_equal(self.conditional_shape, conditional_event_shape)]):

        conditional_sample_shape = conditional_input_shape[
            :conditional_sample_rank]
        repeat = n // prefer_static.reduce_prod(conditional_sample_shape)
        h = tf.reshape(
            conditional_input,
            prefer_static.concat([(-1,), self.conditional_shape], axis=0))
        h = tf.tile(h,
                    prefer_static.pad(
                        [repeat], paddings=[[0, conditional_event_rank]],
                        constant_values=1))

    samples_0 = tf.random.uniform(
        prefer_static.concat([(n,), self.event_shape], axis=0),
        minval=-1., maxval=1., dtype=self.dtype, seed=seed)
    inputs = samples_0 if conditional_input is None else [samples_0, h]
    params_0 = self.network(inputs, training=training)
    samples_0 = self._sample_channels(*params_0, seed=seed)

    image_height, image_width, _ = tensorshape_util.as_list(self.event_shape)
    def loop_body(index, samples):
      inputs = samples if conditional_input is None else [samples, h]
      params = self.network(inputs, training=training)
      samples_new = self._sample_channels(*params, seed=seed)

      # Update the current pixel
      samples = tf.transpose(samples, [1, 2, 3, 0])
      samples_new = tf.transpose(samples_new, [1, 2, 3, 0])
      row, col = index // image_width, index % image_width
      updates = samples_new[row, col, ...][tf.newaxis, ...]
      samples = tf.tensor_scatter_nd_update(samples, [[row, col]], updates)
      samples = tf.transpose(samples, [3, 0, 1, 2])

      return index + 1, samples

    index0 = tf.zeros([], dtype=tf.int32)

    # Construct the while loop for sampling
    total_pixels = image_height * image_width
    loop_cond = lambda ind, _: tf.less(ind, total_pixels)
    init_vars = (index0, samples_0)
    _, samples = tf.while_loop(loop_cond, loop_body, init_vars,
                               parallel_iterations=1)
    return samples

  def _sample_channels(
      self, component_logits, locs, scales, coeffs=None, seed=None):
    num_channels = self.event_shape[-1]

    component_dist = tfd.Categorical(logits=component_logits)
    mask = tf.one_hot(indices=component_dist.sample(seed=seed),
                      depth=self._num_logistic_mix)
    mask = tf.cast(mask[..., tf.newaxis], self.dtype)

    masked_locs = tf.reduce_sum(locs * mask, axis=-2)
    loc_tensors = tf.split(masked_locs, num_channels, axis=-1)
    masked_scales = tf.reduce_sum(scales * mask, axis=-2)
    scale_tensors = tf.split(masked_scales, num_channels, axis=-1)

    if coeffs is not None:
      num_coeffs = num_channels * (num_channels - 1) // 2
      masked_coeffs = tf.reduce_sum(coeffs * mask, axis=-2)
      coef_tensors = tf.split(masked_coeffs, num_coeffs, axis=-1)

    channel_samples = []
    coef_count = 0
    for i in range(num_channels):
      loc = loc_tensors[i]
      if coeffs is not None:
        for c in channel_samples:
          loc += c * coef_tensors[coef_count]
          coef_count += 1
      cb_samp = tfd.ContinuousBernoulli(
          logits=tf.clip_by_value(loc, -15.94, 15.94)).sample(seed=seed)
      channel_samples.append(2*cb_samp - 1)

    return tf.concat(channel_samples, axis=-1)


class PixelCNN(tfk.Model):
  """PixelCNN++ model."""

  def __init__(self, visible_dist, *args, **kwargs):
    super(PixelCNN, self).__init__()
    self.visible_dist = visible_dist
    if visible_dist == 'categorical':
      self.dist = tfd.PixelCNN(*args, **kwargs)
    elif visible_dist == 'cont_bernoulli':
      self.dist = PixelCNNwithCB(*args, **kwargs)
    else:
      raise NotImplementedError

  def call(self, inputs, training=False):
    inputs = tf.reshape(
        inputs, prefer_static.concat([(-1,), self.dist.event_shape], axis=0))
#     inputs = (2. * (inputs - self.dist._low)/(self.dist._high - self.dist._low)) - 1.
    inputs = 2.*inputs - 1
    params = self.dist.network(inputs, training=training)
    num_channels = self.dist.event_shape[-1]
    if num_channels == 1:
      component_logits, locs, scales = params
    else:
      component_logits, locs, scales, coeffs = params
      num_coeffs = num_channels * (num_channels - 1) // 2
      loc_tensors = tf.split(locs, num_channels, axis=-1)
      coef_tensors = tf.split(coeffs, num_coeffs, axis=-1)
      channel_tensors = tf.split(inputs, num_channels, axis=-1)

      coef_count = 0
      for i in range(num_channels):
        channel_tensors[i] = channel_tensors[i][..., tf.newaxis, :]
        for j in range(i):
          loc_tensors[i] += channel_tensors[j] * coef_tensors[coef_count]
          coef_count += 1
      locs = tf.concat(loc_tensors, axis=-1)

    return self.dist._make_mixture_dist(component_logits, locs, scales)

  def nll_loss(self, targets, dist):
    targets = tf.reshape(tf.cast(targets, tf.float32), prefer_static.concat(
        [(-1,), self.dist.event_shape], axis=0))
    return -tf.reduce_mean(dist.log_prob(targets))

  def pixel_ll(self, inputs, targets, training=False):
    # TODO(kushalchauhan): Make this work with n_channels > 1
    inputs = tf.reshape(tf.cast(inputs, tf.float32), prefer_static.concat(
        [(-1,), self.dist.event_shape], axis=0))

    inputs = (2. * (inputs - self.dist._low) /
              (self.dist._high - self.dist._low)) - 1.

    params = self.dist.network(inputs, training=False)

    num_channels = self.dist.event_shape[-1]
    assert num_channels == 1
    component_logits, locs, scales = params

    mixture_distribution = tfd.Categorical(logits=component_logits)
    locs = self.dist._low + 0.5 * (self.dist._high-self.dist._low) * (locs+1.)
    scales *= 0.5 * (self.dist._high - self.dist._low)

    logistic_dist = tfd.QuantizedDistribution(
        distribution=tfd.TransformedDistribution(
            distribution=tfd.Logistic(loc=locs, scale=scales),
            bijector=tfp.bijectors.Shift(shift=tf.cast(-0.5, self.dist.dtype))),
        low=self.dist._low, high=self.dist._high)

    dist = tfd.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=tfd.Independent(
            logistic_dist, reinterpreted_batch_ndims=1))
    return dist.log_prob(tf.cast(targets, tf.float32))

  def compute_corrections(self, dataset=None):
    # pylint: disable=g-long-lambda
    if self.visible_dist == 'categorical':
      assert dataset is not None, ('dataset is required to compute correction '
                                   'for Mixture of Logistics visible '
                                   'distribution.')
      self.corr_dict = collections.defaultdict(list)
      update_dict = lambda corr_dict_img: [
          self.corr_dict[pix].append(
              scipy.special.logsumexp(corr_dict_img[pix])-np.log(
                  len(corr_dict_img[pix]))) for pix in corr_dict_img
      ]
      j = 0
      for train_batch in tqdm.tqdm(dataset):
        j += 1
        pixel_ll = self.pixel_ll(*train_batch)
        inp = train_batch[1].numpy().squeeze()
        if self.dist.event_shape[-1] == 3:
          inp[:, :, :, 1:] += 256
          inp[:, :, :, 2:] += 256
        for i in range(inp.shape[0]):
          corr_dict_img = collections.defaultdict(list)
          np.vectorize(lambda pix, ll: corr_dict_img[int(pix)].append(ll),
                       otypes=[float])(inp[i], pixel_ll[i])
          update_dict(corr_dict_img)
        if j == 500:
          break

      for key in self.corr_dict:
        n = len(self.corr_dict[key])
        self.corr_dict[key] = scipy.special.logsumexp(
            np.array(self.corr_dict[key])) - np.log(n)
      f = scipy.interpolate.interp1d(
          *list(zip(*[(pix, corr) for pix, corr in self.corr_dict.items()])),
          fill_value='extrapolate')
      for missing_pix in (set(range(256 * self.dist.event_shape[-1])) -
                          set(self.corr_dict)):
        self.corr_dict[missing_pix] = f(missing_pix)
      self.correct = np.vectorize(lambda x: self.corr_dict[x])
    elif self.visible_dist == 'cont_bernoulli':
      self.corr_dict = {}
      targets = np.round(np.linspace(1e-3, 1-1e-3, 999), decimals=3)
      for target in targets:
        self.corr_dict[target] = -utils.cb_neglogprob(scipy.optimize.fmin(
            utils.cb_neglogprob, 0.5, args=(target,), disp=False)[0], target)

      corr_func = lambda pix: self.corr_dict[(np.clip(pix, 1e-3, 1-1e-3)
                                              .astype(float)
                                              .round(decimals=3)
                                             )].astype(np.float32)

      self.correct = np.vectorize(corr_func)
    # pylint: enable=g-long-lambda
