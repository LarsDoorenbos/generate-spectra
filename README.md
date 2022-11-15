# Generating galaxy spectra from photometry with conditional diffusion models

Code used for our paper ["Generating astronomical spectra from photometry with conditional diffusion models"](https://arxiv.org/abs/2211.05556), accepted at the NeurIPS 2022 workshop on Machine Learning and the Physical Sciences.

![Some generated spectra](figures/generated.png "samples")

This repository contains the code to train our contrastive and generative networks. Code to download the data is provided in the `utils` folder

### Abstract

A trade-off between speed and information controls our understanding of astronomical objects. Fast-to-acquire photometric observations provide global properties, while costly and time-consuming spectroscopic measurements enable a better understanding of the physics governing their evolution. Here, we tackle this problem by generating galaxy spectra directly from photometry, through which we obtain an estimate of their intricacies from easily acquired images. This is done by using multimodal conditional diffusion models, where the best out of the generated spectra is selected with a contrastive network. Initial experiments on minimally processed SDSS data show promising results.

### Citation

If you find this work helpful, consider citing it using

```
@article{doorenbos2022generating,
  title={Generating astronomical spectra from photometry with conditional diffusion models},
  author={Doorenbos, Lars and Cavuoti, Stefano and Longo, Giuseppe and Brescia, Massimo and Sznitman, Raphael and M{\'a}rquez-Neila, Pablo},
  journal={arXiv preprint arXiv:2211.05556},
  year={2022}
}
```
