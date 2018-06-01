from keras import models, Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization

def create_model():
    img_w = 64
    img_h = 64
    n_labels = 1

    kernel = 3

    encoding_layers = [
        Convolution2D(64, kernel, padding='same', input_shape=( img_h, img_w,3)),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]

    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
        #print(l.input_shape,l.output_shape,l)

    decoding_layers = [
        UpSampling2D(),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(n_labels, 1, padding='valid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    # autoencoder.add(Reshape((n_labels, img_h * img_w)))
    # autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('sigmoid'))

    return autoencoder

from keras import backend as K
from keras.layers import concatenate
from build_ae import create_models, loader

def main():
    encoder, _, vae = create_models()
    vae.load_weights('encoder.008.h5')
    encoder.trainable = False

    seg = create_model()

    x = Input(shape=(64, 64, 3), name='input_image')
    mask = seg(x)

    y = concatenate([x, mask])

    z_mean, z_log_var = encoder(y)

    kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))

    model = Model(x, [z_mean, z_log_var])
    model.add_loss(kl_loss)

    model.compile('nadam')
    model.load_weights('segnet.010.h5')

    model.summary()

    ck = ModelCheckpoint('segnet.{epoch:03d}.h5', save_weights_only=True)

    data = loader(False)
    # model.fit_generator(data, 100, 10, callbacks=[ck])


    # Test code


    samples = 5

    data = loader(True, samples)

    import matplotlib.pyplot as plt

    d,gt = next(data)
    d = d[:, :, :, :3]
    q = seg.predict_on_batch(d)

    for i in range(len(d)):
        p = i * 3
        plt.subplot(samples, 3, p + 1)
        plt.imshow(d[i].squeeze(), cmap='gray')

        plt.subplot(samples, 3, p + 2)
        plt.imshow(gt[i].squeeze(), cmap='gray')

        plt.subplot(samples, 3, p + 3)
        plt.imshow(q[i].squeeze(), cmap='gray')

    plt.show()




if __name__ == '__main__':
    main()
