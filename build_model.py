import glob
from multiprocessing.pool import ThreadPool

from PIL import Image
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
from build_ae import create_models
import numpy as np

def _load_image(f):
    im = Image.open(f) \
              .crop((0, 20, 178, 198)) \
              .resize((64, 64), Image.BICUBIC)
    return np.asarray(im)


def loader(encoder_train=True, batch_size=64, normalize=True):
    ball_img = Image.open('ball.png', 'r')
    files = glob.glob('/home/darwin/vaegan-celebs-keras/img_align_celeba_png/*.png')
    with ThreadPool(1) as p:
        while True:
            np.random.shuffle(files)

            for s in range(0, len(files), batch_size):
                e = s + batch_size
                batch_names = files[s:e]
                batch_images = p.map(_load_image, batch_names)
                batch_images = np.stack(batch_images)

                if normalize:
                    batch_images = batch_images / 255.
                    # To be sure
                    #batch_images = np.clip(batch_images, -1., 1.)

                bs = len(batch_images)
                ones = np.zeros((bs, 64, 64, 1))
                images = np.concatenate([batch_images, ones], -1)
                #images = np.random.uniform(-1., 1., size=(batch_size, 64, 64, 4))

                # Clear mask
                #images[:, :, :, -1] = -1.


                for i in range(len(images)):
                    s = np.random.randint(24, 48)
                    img = np.asarray(ball_img.resize((s, s), Image.BICUBIC)) / 255.
                    x = np.random.randint(0, 64 - s)
                    y = np.random.randint(0, 64 - s)
                    # images[i, :, :, :3] = batch_images[i]
                    images[i, y:y+s, x:x+s, :] = img

                mask = (np.expand_dims(images[:, :, :, -1], -1) )#+ 1.) / 2.

                if encoder_train:
                    yield images, mask
                else:
                    yield images[:, :, :, :3], None


def main():
    encoder, _, vae = create_models()
    encoder.load_weights('encoder-trained.h5')
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
    # model.load_weights('segnet.010.h5')

    model.summary()

    ck = ModelCheckpoint('segnet.{epoch:02d}.h5', save_weights_only=True)

    data = loader(False)
    model.fit_generator(data, 1000, 100, callbacks=[ck])


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
