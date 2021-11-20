# Weather-Classifier

This repo contains code and models trained to classify images, which contains daytime, night, kinds of weather such as snows, rains, sunny, etc.

### Requirements
- [numpy](https://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [OpenCV 3](https://pypi.org/project/opencv-python/3.4.9.31/)
- [Pytorch](https://pytorch.org/get-started/locally/)
- [torchvision](https://pytorch.org/get-started/locally/)
- [seaborn](https://seaborn.pydata.org/index.html)

### Dataset
The data for this project was from [ACDC](https://acdc.vision.ee.ethz.ch/download) using the [rgb_anon_trainvaltest](https://acdc.vision.ee.ethz.ch/rgb_anon_trainvaltest.zip) datasets.

But the size of the official data is large, you can download the sample data corresponding to the code with this link [sample data](). (Just as an example, to check the how the code runs)

There are 4006 8-bit RGB images about Anonymized adverse-condition images for train, val and test sets, and simultaneously 4006 images corresponding anonymized normal-condition images.

### Models

Two different approaches have been used.

- Baseline model - Basic model that uses average brightness from Value channel of HSV image as threshold, with 3 filters from Hue Channel of HSV, R Channel and G Channel of RGB,  to classify images. <br/>Achieves an accuracy of $98.56\%$ on the validation set and $94.40\%$ on test set.
- Simple CNN - A Simple 5-layer Fully Convolutional Neural Network, that works on Value channel of HSV image. <br/>Achieves an accuracy of $100\%$ on the validation set.

### Files

#### Night / Day Classification

##### 1. Training

- [night_baseline.ipynb](./training/night_baseline.ipynb) - Training of baseline model according to brightness.
- [night_cnn.ipynb](./training/night_cnn.ipynb) - Training of Simple 5-layer CNN model
- [night_baseline.py](./training/night_baseline.py) - Perform baseline prediction on image using HSV and RGB thresholds.

##### 2. Utilis

- [DataLoader.py](./utils/day_night/DataLoader.py) - Load the images, combine path, HSV and RGB value of each image as a dataframe.
- [Estimation.py](./utils/day_night/Estimation.py) - Find the best Value, Hue, Red and Green threshold, as well as the maximal accuracy.
- [Visualization.py](./utils/day_night/Visualization.py) - Visualize the images with the V, H, R, G channel value, scatter plots or real images.

#### Syntax for inference

```bash
git clone git@github.com:LonelVino/Weather_classification.git
cd Weather_classification
python3 ./training/night_baseline.py
```

#### Sample Results

> ### Inference
>
> - [predict_simple_model.py](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/predict_simple_model.py) - Perform prediction on image using the Simple 5-layer CNN
> - [predict_mbv2.py](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/predict_mbv2.py) - Perform prediction on image using the MobileNetv2 model.
> - [predict_all_models.py](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/predict_all_models.py) - Performs prediction on image using all 3 models and outputs the results side by side for comparision.
>
> #### Syntax for inference
>
> ```
> python predict_file.py -i /path/to/image.jpg
> ```
>
> Example:
>
> ```
> python predict_all_models.py -i day_night_dataset/val/night/pexels-photo-2403202.jpeg
> ```
>
> #### Sample ResultsNote: The data below is obtained by training **the whole dataset**, different with the results of the sample data. 

| Threshold          | Train Accuracy | Validation Accuracy | Test Accuracy |
| ------------------ | -------------- | ------------------- | ------------- |
| $\mathrm{V}$       | $92.50\%$      | $97.64\%$           | $89.60\%$     |
| $\mathrm{V+H}$     | $94.00\%$      | $98.56\%$           | $92.00\%$     |
| $\mathrm{V+H+R+G}$ | $95.38\%$      | $98.56\%$           | $94.40\%$     |

Where, $\mathrm{V,H}$ mean Value and Hue Channels of HSV, and $\mathrm{R,G}$ mean Red and Green Channels of RGB.

| Threshold          | $\mathrm{V}$ Channel | $\mathrm{H}$ Channel | $\mathrm{R}$ Channel | $\mathrm{G}$ Channel |
| ------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| **Train Set**      | $80.70$              | $23.00$              |                      |                      |
| **Validation Set** | $79.00$              | $27.00$              |                      |                      |
| **Test Set**       | $80.70$              | $23.00$              |                      |                      |



| True Label | Predict Label | Condition (Feature) | Correct |
| ---------- | ------------- | ------------------- | ------- |
| Night      | Day           | H < $H_{\min}$      | Night   |
| Day        | Night         | R > $R_{\max}$      | Day     |
| Night      | Day           | G > $G_{\max}$      | Night   |

