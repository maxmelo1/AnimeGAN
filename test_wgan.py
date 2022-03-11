import os
from matplotlib import pyplot as plt
import numpy as np
import PIL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from model.wgan import build_generator, build_critic, WGAN, GANMonitor, d_wasserstein_loss, g_wasserstein_loss

import argparse

print(tf.version.VERSION)

LR = 0.00005 # UPDATE for WGAN: learning rate per WGAN paper
LATENT_DIM = 128 
BATCH_SIZE = 64

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--weight', type=str, help='trained weight path')


args = parser.parse_args()

model_path = args.weight
model = keras.models.load_model(model_path)

model.compile()

x = tf.random.normal(shape=(BATCH_SIZE, LATENT_DIM))

def show(images):
    plt.figure(figsize=(4, 4))

    bs = images.shape[0]

    for i in range(0, bs, 16):
        for j in range(16):
            plt.subplot(4, 4, j+1)
            img = keras.utils.array_to_img(images[i+j])
            plt.imshow(img)
            plt.axis('off')
        plt.show()

res = model.predict(x)

show(res)
# plt.axis("off")
# plt.imshow(res[0])
# plt.show()