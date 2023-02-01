# UNIT-DDPM-Unofficial

Unofficial "partial" implementation of https://arxiv.org/abs/2104.05358

## Overview
UNIT-DDPM is an unpaired image-to-image translation method using diffusion models and tweaks in sampling strategy. This github repo aims to implement the training and sampling methods used in the paper with contents from the original DDPM.

## Usage
`train.py` and `sampling.py` contains codes for the training and sampling strategy, respectively. Edit your desired paths and params in `configs.py` first. An alternate way would be using the provided jupyter template instead. I have trained the models on an anime-to-human cycleGAN dataset, which explains my variable names.

## Acknowledgement
Thank you https://github.com/lucidrains/denoising-diffusion-pytorch for the original diffusion codes
