'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



class VAE:

    def __init__(self, original_dim = 28 ** 2,
        intermediate_dim = 512, batch_size = 128, latent_dim = 2, epochs = 50):
        self.original_dim = original_dim
        self.input_shape = (original_dim, )
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.decoder = None
        self.encoder = None

    def compile(self):
        # VAE model = encoder + decoder
        # build encoder model

        original_dim = self.original_dim
        input_shape = self.input_shape
        intermediate_dim = self.intermediate_dim
        batch_size = self.batch_size
        latent_dim = self.latent_dim
        epochs = self.epochs

        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)


        ### KEY DIFFERENCE WITH NORMAL AUTOENCODERS ###
        # use Reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        # Set loss and compile
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= self.original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()

        self.encoder = encoder
        self.decoder = decoder
        return vae

    def plot_results(self,
                     data,
                     batch_size=128,
                     model_name="vae_mnist"):
        """Plots labels and MNIST digits as function of 2-dim latent vector

        # Arguments:
            models (tuple): encoder and decoder models
            data (tuple): test data and label
            batch_size (int): prediction batch size
            model_name (string): which model is using this function
        """

        encoder, decoder = self.encoder, self.decoder
        x_test, y_test = data
        os.makedirs(model_name, exist_ok=True)

        filename = os.path.join(model_name, "vae_mean.png")
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = encoder.predict(x_test,
                                       batch_size=batch_size)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig(filename)
        plt.show()

        filename = os.path.join(model_name, "digits_over_latent.png")
        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        plt.show()

    def plot_results_1D(self, smin=-4, smax=4, r=10, image_dim=(28, 28)):
        """
        Plot results from 1D encoding in a square. Sample from 1D interval and show progression.
        data (tuple): test data and label
        :param smin: Minimum latent parameter.
        :param smax: Maximum latent parameter.
        :param r: Number of images in the square is r**2.
        :param image_dim: Original dimensions of the images.
        :return:
        """
        encoder, decoder = self.encoder, self.decoder
        h, w = image_dim
        n = r ** 2
        grid_x = np.linspace(smin, smax, n)
        figure = np.zeros((h * r, w * r))
        for i, j in it.product(range(r), range(r)):
            t = r * i + j
            x_decoded = decoder.predict(np.array([grid_x[t]]))
            img = x_decoded[0].reshape(h, w)
            figure[i * h: (i + 1) * h,
                   j * w: (j + 1) * w] = img
        plt.figure()
        plt.imshow(figure, cmap='Greys_r')
        plt.show()