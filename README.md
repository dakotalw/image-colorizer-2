# [Title]
By Jonathan Srinivasan and Dakota Wilson                                    
DS-5899: Special Topics in Data Science - Transformers in Theory and Practice  

## Overview

**Problem:** 


## Data

- Dataset of sports photos (408.5 MB)
- Includes photos of over 100 sports, with 10,000+ entries
- https://www.kaggle.com/datasets/gpiosenka/sports-classification


## Methodology

### Convolutional Neural Network
- Built a model that 1) decomposes a color picture and recolorizes it and 2) colorizes black and white photos
- Improved upon a pretrained model (ex: switched loss function from MSE to LPIPS)
- 18 layers that reduce and convolute the photo, and then deconvolutes it back to a full size image

![Capture1](https://user-images.githubusercontent.com/48261978/231642125-66ce5354-792b-4f5d-a11e-699f3e4beacd.PNG)

- LPIPS: Learned Perceptual Image Patch Similarity - a metric created to try and replicate human vision. This loss function takes in two images and tries to define how different they are in the sense of human vision, as opposed to solely relying on pixel-wide differences. The LPIPS loss is computed as the cosine distance between the feature representations of the source and target images at each layer, normalized by the magnitude of the feature vectors. In our CNN project, we had taken a pre-existing model and outfitted it with this new loss to improve upon original results. This is the metric that we will be using in this project for evaluating the accuracy of our outputs against each other. 

### Google ColTran

- Why use a transformer for this when a CNN architecture has proven to be a powerful learner that is able to do a decent job of recolorizing an image?

Attention can prove to be incredibly helpful in the completion of this task; by increasing the receptive field that the model is able to "see" at each step, we expect our model to be able to provide more accurate results that take in the context of the whole photo. This context is something that a CNN is simply not able to find by taking traditional strides across an image. 

### Stable Diffusion


### Testing


### Limitations

Unfortunately, a common issue experienced across all computer vision tasks regardless of architecture is the high computational cost of processing images. This issue only worsens with higher quality images, which are becoming more and more prevalent as time goes on. For both our model and ColTran, images needed to be resized to 224 x 224 pixels for computational efficiency. This increase in time and computation experienced with larger images may be manageable when we are predicting across images, but training a model with larger images will quickly add orders of magnitude to computation time and costs.

This is not the worst problem to have, however! Modern transformer architectures have been developed to upscale images from lower pixel dimensions into higher quality images. This is even applied in ColTran; in the third and final step of predicting, the model takes a 64 x 64 image and scales it all the way up to 224 x 224.

## Results/Analysis


## Resources

- https://github.com/richzhang/colorization
- https://www.kaggle.com/code/theblackmamba31/autoencoder-grayscale-to-color-image
- https://lukemelas.github.io/image-colorization.html (Baseline model)

