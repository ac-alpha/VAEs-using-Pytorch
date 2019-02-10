<img src="images/digit_transition.png" width="200" /> <img src="images/scatter_both.png" width="250" />

# Variational Auto-Encoders using Pytorch

` Variational Auto-Encoder Implementation trained on MNIST Dataset in Pytorch`

## Introduction

Variational AutoEncoders are a class of Generative Models which are used to deal with models of distributions `P(X)`, defined over datapoints X in some potentially high-dimensional space X. We get examples X distributed according to some unknown distribution `Pgt(X)`, and our goal is to learn a model P which we can sample from, such that `P` is as similar as possible to `Pgt`.

First we map our original image X to a latent variable z using some distribution `Q(z|X)` . Then we pass that value of z to the distribution `P(X|z)` to get an image as close to original image X.
Before we can say that our model is representative of our dataset, we need to make sure that for every datapoint X in the dataset, there is one (or many) settings of the latent variables which causes the model to generate something very similar to X. 

We wish to optimize parameters such that we can sample z from `P(z)` and, with high probability, `P(X|z)` will be like the X’s in our dataset. For most z, `P(X|z)` will be nearly zero, and hence contribute
almost nothing to our estimate of `P(X)`. The key idea behind the variational
autoencoder is to attempt to sample values of z that are likely to have
produced X, and compute `P(X)` just from those.

## Pipeline

<img src="images/VAE_pipeline.png" />

This image has been taken from this [tutorial](https://arxiv.org/pdf/1606.05908).
