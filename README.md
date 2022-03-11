# Anime GAN
Simple implementation of Anime GAN and Anime GAN-GP of [Pyimagesearch](https://pyimagesearch.com/2022/02/07/anime-faces-with-wgan-and-wgan-gp/).

This implementation simply aims to generate anime faces using WGAN and WGAN-GP, just for fun :).

The train_wgan.py script can be used to train the model from scratch. The test_wgan.sh/py is used to generate a sample from the previously trained model. Some examples are shown below.

The models were trained with 50 epochs each with learning rate of 0.00005. The trained models are made available to download. Feel free to fine tune them to achieve even better visual results.

## WGAN generated examples
<img src="imgs/output0.png" alt="Example 1" width="200"/>

<img src="imgs/output1.png" alt="Example 2" width="200"/>

## WGAN-GP generated examples
<img src="imgs/gp_output0.png" alt="Example 1" width="200"/>

<img src="imgs/gp_output1.png" alt="Example 2" width="200"/>