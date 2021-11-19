# Weather-Classifier
This repo contains code and models trained to classify images, which contains daytime, night, kinds of weather such as snows, rains, sunny, etc.

### Requirements
- [numpy](https://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [OpenCV 3](https://pypi.org/project/opencv-python/3.4.9.31/)
- [Pytorch](https://pytorch.org/get-started/locally/)
- [torchvision](https://pytorch.org/get-started/locally/)

### Dataset
The data for this project was from [ACDC](https://acdc.vision.ee.ethz.ch/download) using the [rgb_anon_trainvaltest](https://acdc.vision.ee.ethz.ch/rgb_anon_trainvaltest.zip) datasets.

But the size of the official data is large, you can download the sample data corresponding to the code with this link[sample data](). (Just as an example, to check the how the code runs)

There are 4006 8-bit RGB images about Anonymized adverse-condition images for train, val and test sets, and simultaneously 4006 images corresponding anonymized normal-condition images.

### Models

### Files
#### Night / Day Classification
- [night_baseline.ipynb](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/training/baseline.ipynb) - Training of baseline model according to brightness.
- [night_cnn.ipynb](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/training/simple_hsv_model.ipynb) - Training of Simple 5-layer CNN model