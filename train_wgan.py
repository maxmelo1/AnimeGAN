import os
from matplotlib import pyplot as plt
import numpy as np
import PIL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from model.wgan import build_generator, build_critic, WGAN, GANMonitor, d_wasserstein_loss, g_wasserstein_loss

print(tf.version.VERSION)

def show(images):
    plt.figure(figsize=(4, 4))

    for i in range(16):
        plt.subplot(4, 4, i+1)
        img = keras.utils.array_to_img(images[i])
        plt.imshow(img)
        plt.axis('off')
    plt.show()

anime_data_dir = "dataset/images/"

train_images = tf.keras.utils.image_dataset_from_directory(
   anime_data_dir, label_mode=None, image_size=(64, 64), batch_size=64)

image_batch = next(iter(train_images))
random_index = np.random.choice(image_batch.shape[0])
random_image = image_batch[random_index].numpy().astype("int32")
plt.axis("off")
plt.imshow(random_image)
plt.show()

train_images = train_images.map(lambda x: (x - 127.5) / 127.5)

show(image_batch[:16])

# latent dimension of the random noise
LATENT_DIM = 128 
# weight initializer for G per DCGAN paper 
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) 
# number of channels, 1 for gray scale and 3 for color images
CHANNELS = 3



generator = build_generator(WEIGHT_INIT, LATENT_DIM, CHANNELS)
generator.summary()



# build the critic model
critic = build_critic(64, 64, 3)
critic.summary()



wgan = WGAN(critic=critic, 
              generator=generator, 
              latent_dim=LATENT_DIM,
              critic_extra_steps=5) # UPDATE for WGAN



LR = 0.00005 # UPDATE for WGAN: learning rate per WGAN paper

wgan.compile(
    d_optimizer = keras.optimizers.RMSprop(learning_rate=LR, clipvalue=1.0, decay=1e-8), # UPDATE for WGAN: use RMSProp instead of Adam
    g_optimizer = keras.optimizers.RMSprop(learning_rate=LR, clipvalue=1.0, decay=1e-8), # UPDATE for WGAN: use RMSProp instead of Adam
    d_loss_fn = d_wasserstein_loss,
    g_loss_fn = g_wasserstein_loss
)

NUM_EPOCHS = 100 # number of epochs
wgan.fit(train_images, epochs=NUM_EPOCHS, callbacks=[GANMonitor(num_img=16, latent_dim=LATENT_DIM)])