# Image (Re)Colorization v2: Transformer Edition
By Jonathan Srinivasan and Dakota Wilson                                    
DS-5899: Special Topics in Data Science - Transformers in Theory and Practice  

## Overview

**Problem:** What performs better in colorizing black and white photos? CNNs or Google ColTran?
*Note: How do we calculate image coloring performace? Loss function? Eye test?*


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

![ColTran Architecture](https://github.com/dakotalw/image-colorizer-2/blob/main/coltran_architecture.png)

ColTran is comprised of three parts: an autoregressive colorizer that is based on a 


### Testing

![outputs](https://github.com/dakotalw/image-colorizer-2/blob/main/grid_of_images.png)

In our testing, the CNN output images actually had a slightly lower LPIPS score than ColTran. There could be several reasons for this, but it is more than likely due to the precision in the color of ColTran. We can observe that although the colors may not be as accurate, ColTran does a much better job of being consistent with its color choices, compared to the patchy look that our CNN gives. If an object is supposed to be blue, and a model predicts it as half blue and half red, the LPIPS would be lower for that prediction than if it had predicted completely red. However, when viewing one of these images, the all red image would look more natural to the human eye. This demonstrates both the weaknesss of LPIPS as well as how difficult it can be to create a loss function that accurately represents human vision.

### Limitations

Unfortunately, a common issue experienced across all computer vision tasks regardless of architecture is the high computational cost of processing images. This issue only worsens with higher quality images, which are becoming more and more prevalent as time goes on. For both our model and ColTran, images needed to be resized to 224 x 224 pixels for computational efficiency. This increase in time and computation experienced with larger images may be manageable when we are predicting across images, but training a model with larger images will quickly add orders of magnitude to computation time and costs.

This is not the worst problem to have, however! Modern transformer architectures have been developed to upscale images from lower pixel dimensions into higher quality images. This is even applied in ColTran; in the third and final step of predicting, the model takes a 64 x 64 image and scales it all the way up to 224 x 224.

## Results/Analysis


## Stable Diffusion

- Neural Network called ControlNet
- **Canny Edge Detection Algorithm:** applied to the image to extract edges and convert it into a binary image. The process involves smoothing the image using a Gaussian filter, computing the gradient magnitude and direction at each pixel, suppressing non-maximum edges, and hysteresis thresholding to remove weak edges.        
- **Memory-Efficient Attention:** uses local attention, where the input sequence is split into chunks and self attention is applied to the chunks within a certain distance of the current position. This reduces the number of computations and memory required to compute the self-attention scores, and allows the model to handle longer input sequences.  

```python
enable_xformers_memory_efficient_attention()
```                                    

![eating](https://user-images.githubusercontent.com/48261978/231761743-e9a30b62-1fde-428b-9830-e9196ee9f5aa.jpg)

Prompt: "Barack Obama eating food. Barack Obama is happy to be eating food. High quality"                                                
Negative Prompt: "black and white, low quality, no food"                                  

![download](https://user-images.githubusercontent.com/48261978/231765686-463a073a-ead1-4e49-99e7-812389570b1d.png)

![obama1](https://user-images.githubusercontent.com/48261978/231761802-90e5ee0c-db74-4197-8c95-f5530490297a.png)


## Resources

- https://github.com/richzhang/colorization
- https://www.kaggle.com/code/theblackmamba31/autoencoder-grayscale-to-color-image
- https://lukemelas.github.io/image-colorization.html (Baseline model)

