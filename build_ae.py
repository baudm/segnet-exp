#!/usr/bin/env python3

import numpy as np

from keras import Input, Model, Sequential
from keras.layers import BatchNormalization, Conv2DTranspose, LeakyReLU, Conv2D, Activation, Flatten, Dense, Reshape, \
    Lambda
from keras import backend as K


def create_models():
    n_channels = 3 + 1
    image_shape = (64, 64, n_channels)
    n_encoder = 1024
    latent_dim = 128
    decode_from_shape = (8, 8, 256)
    n_decoder = np.prod(decode_from_shape)
    leaky_relu_alpha = 0.2

    def conv_block(x, filters, leaky=True, transpose=False, name=''):
        conv = Conv2DTranspose if transpose else Conv2D
        activation = LeakyReLU(leaky_relu_alpha) if leaky else Activation('relu')
        layers = [
            conv(filters, 5, strides=2, padding='same', name=name + 'conv'),
            BatchNormalization(name=name + 'bn'),
            activation
        ]
        if x is None:
            return layers
        for layer in layers:
            x = layer(x)
        return x


    # Encoder
    def create_encoder():
        x = Input(shape=image_shape, name='enc_input')

        y = conv_block(x, 64, name='enc_blk_1_')
        y = conv_block(y, 128, name='enc_blk_2_')
        y = conv_block(y, 256, name='enc_blk_3_')
        y = Flatten()(y)
        y = Dense(n_encoder, name='enc_h_dense')(y)
        y = BatchNormalization(name='enc_h_bn')(y)
        y = LeakyReLU(leaky_relu_alpha)(y)

        z_mean = Dense(latent_dim, name='z_mean')(y)
        z_log_var = Dense(latent_dim, name='z_log_var')(y)

        return Model(x, [z_mean, z_log_var], name='encoder')


    # Decoder
    decoder = Sequential([
        Dense(n_decoder, input_shape=(latent_dim,),
              name='dec_h_dense'),
        BatchNormalization(name='dec_h_bn'),
        LeakyReLU(leaky_relu_alpha),
        Reshape(decode_from_shape),
        *conv_block(None, 256, transpose=True, name='dec_blk_1_'),
        *conv_block(None, 128, transpose=True, name='dec_blk_2_'),
        *conv_block(None, 32, transpose=True, name='dec_blk_3_'),
        Conv2D(1, 5, activation='sigmoid', padding='same', name='dec_output')
    ], name='decoder')


    def _sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
           Instead of sampling from Q(z|X), sample eps = N(0,I)

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


    sampler = Lambda(_sampling, output_shape=(latent_dim,), name='sampler')

    encoder = create_encoder()

    # Build graph
    x = Input(shape=image_shape, name='input_image')

    z_mean, z_log_var = encoder(x)
    z = sampler([z_mean, z_log_var])

    y = decoder(z)

    vae = Model(x, y, name='vae')

    # KL divergence loss
    kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
    vae.add_loss(kl_loss)

    return encoder, decoder, vae


from PIL import Image


def loader(encoder_train=True, batch_size=64):
    ball_img = Image.open('ball.png', 'r')
    while True:
        images = np.random.uniform(size=(batch_size, 64, 64, 4))
        # Clear mask
        images[:, :, :, -1] = 0.
        for i in range(len(images)):
            s = np.random.randint(24, 48)
            img = np.asarray(ball_img.resize((s, s), Image.BICUBIC)) / 255.
            x = np.random.randint(0, 64 - s)
            y = np.random.randint(0, 64 - s)
            images[i, y:y+s, x:x+s, :] = img

        mask = np.expand_dims(images[:, :, :, -1], -1)

        if encoder_train:
            yield images, mask
        else:
            yield images[:, :, :, :3], None


from keras.callbacks import ModelCheckpoint


def main():
    encoder, decoder, vae = create_models()
    datagen = loader()

    ck = ModelCheckpoint('encoder.{epoch:03d}.h5', save_weights_only=True)

    vae.compile('nadam', 'binary_crossentropy', ['acc'])

    vae.fit_generator(datagen, 1000, 100, callbacks=[ck])
    encoder.save_weights('encoder-trained.h5')




if __name__ == '__main__':
    main()
