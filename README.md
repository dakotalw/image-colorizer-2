# DS5660: Modeling and Machine Learning 2 Project

## Presentations
### Project Proposal
- https://docs.google.com/presentation/d/1-sJxK8-0W_SLHYYUefFSw8JdowlFc3I6kJrkmOeK_As/edit?usp=sharing
### Project Update
- https://docs.google.com/presentation/d/1Xwd8CInpiFikdChDsGNFEL_u_XBHfl_sn9EOeuunm7Q/edit#slide=id.g1764e961685_0_0
### Final Report
- https://docs.google.com/presentation/d/1cXnm0U4syPZBJmaWw4hSOy2qYJXzOz-uaxTMBYvSuJs/edit#slide=id.g1a69b522519_0_665

## Dataset
- https://www.kaggle.com/datasets/gpiosenka/sports-classification

## Sample Code
- https://github.com/richzhang/colorization
- https://www.kaggle.com/code/theblackmamba31/autoencoder-grayscale-to-color-image
- https://lukemelas.github.io/image-colorization.html (Baseline model)

## PyTorch Help
- https://app.datacamp.com/learn/courses/deep-learning-with-pytorch

## Team
- Donald Kane (donald.j.kane@vanderbilt.edu)
- Dakota Wilson (dakota.l.wilson@vanderbilt.edu)
- Jonathan Srinivasan (jonathan.devaraj.srinivasan@vanderbilt.edu)

## Instructions to run:
- Change loaded .pth file path in run-script.py to where 'model-epoch-186-losses-0.124.pth' (found in checkpoints folder) is saved locally. 

- Python script will take in one parameter which is the path for image to be colorized. 

- Output will show the original image, the image with all color removed (model input), and the re-colorized image (model output).

- Example input: python final-script.py /path/on/your/machine/image.jpeg
