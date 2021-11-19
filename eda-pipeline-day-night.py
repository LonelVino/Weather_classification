#!/usr/bin/env python
# coding: utf-8

from utils.day_night import *

import numpy as np
import pandas as pd
import math

import seaborn as sns
import matplotlib.pyplot as plt
import random

# Define the Visualization functions of misclassified images
# - `visualize_mis_images`: Visualize the real images.
# - `scatter_plot_mis_images`: show the misclassified images in a scatter plot.
    
def visualize_mis_images(mis_images):
    ''' Visualize misclassified example(s), show the true label / brigtness / predicted label
    
    Args:
        mis_images (pd.DataFrame): the misclassified images
    '''
    len_images = len(mis_images) if len(mis_images)<=25 else 25
    num = math.ceil(math.sqrt(len_images))
    idxs = random.sample(range(0, len(mis_images)), len_images)
    
    fig = plt.figure(figsize=(num**2,num**2)) if num > 3 else  plt.figure(figsize=((num+1)**2,(num+1)**2))
    plt.title("Misclassified images (True Label - Brightness - Predicted Label)",  fontsize=24)
    for count, index in enumerate(idxs):
        ax = fig.add_subplot(num, num, count + 1, xticks=[], yticks=[])
        image = mis_images.iloc[index].img
        label_true = mis_images.iloc[index].true_label
        label_pred = mis_images.iloc[index].pred_label
        bright = mis_images.iloc[index]['avg_b'] if 'avg_b' in mis_images else  mis_images.iloc[index]['avg_V'] 
        ax.imshow(image)
        ax.set_title("{} {:0.0f} {}".format(label_true, bright, label_pred))

        if count==len_images-1:
            break
        
        
def scatter_plot_mis_images(MISCLASSIFIED_avg_hsv, x_name, y_name, label_name, mode, loc_legend='lower left'):
    ''' Visualize misclassified example(s) with scatter plot, and print out the accuracy of classification
    
    Args:
        MISCLASSIFIED_avg_hsv (pd.DataFrame): the misclassified images with HSV and RGB value
        x_name (String): the name of X_axis, also the field of dataframe, such as `avg_H`, `avg_R`
        y_name (String): the name of Y_axis, also the field of dataframe, such as `avg_H`, `avg_R`
        label_name (String): the label of images, such as `true_label`, `pred_label`
        mode (String): 3 choices -- train / validation / test mode
        loc_legend (String): the location of the legend of the plot
    ''' 
    ax = sns.scatterplot(x=x_name, y=y_name, hue=label_name, data=MISCLASSIFIED_avg_hsv, legend='full')
    ax.legend(loc=loc_legend)
    ax.set_title('[{:s} -- {:s}] of MISCLASSIFED Images ({:s})'.format(x_name, y_name, mode), fontsize=20)
    length = len_train if mode=='train' else len_val if mode=='val' else len_test
    accuracy = (1-len(MISCLASSIFIED_avg_hsv)/ length)*100
    print('The accuracy is: {:.2f} %'.format(accuracy))
    plt.show()

def scatter_rgb(df, x_name, label_name, mode, loc_legend, title, hlines=None, is_mis=False):
    ''' Scatter plot of images in (1 * 3) subplots
        X axis is a Channel Value selected from HSV and RGB,
        Y axis is R Channel, G Channel, B Channel respectively in each subplot
        
    Args:
        df (pd.DataFrame): the misclassified images with HSV and RGB value
        x_name (String): the name of X_axis, also the field of dataframe, such as `avg_H`, `avg_R`
        label_name (String): the label of images, such as `true_label`, `pred_label`
        mode (String): 3 choices -- train / validation / test mode
        loc_legend (String): the location of the legend of the plot
        title (String): the title of the figure (not subplot)
        hlines (list of float): the position of horizontal line in each subplot. Default None
        is_mis (Boolean): the condition of calculating the accuracy of classification
                if True, calculate the accuracy and print it in the terminal 
    '''
    f = plt.figure(figsize=(20,5))
    rgb = ['avg_R', 'avg_G', 'avg_B']
    f.suptitle('{:s} (Mode: {:s})'.format(title, mode), fontsize=20)
    for i in range(3):
        ax = f.add_subplot(1, 3 ,i+1)
        ax = sns.scatterplot(x=x_name, y=rgb[i], hue=label_name, data=df, legend='full')
        ax.legend(loc=loc_legend)
        if hlines != None: ax.axhline(hlines[i], color='red')
        ax.set_title('[{:s} -- {:s}]'.format(x_name, rgb[i]), fontsize=12)
    
    plt.show()
    
    if is_mis:
        length = len_train if mode=='train' else len_val if mode=='val' else len_test
        accuracy = (1-len(df)/ length)*100
        print('The accuracy is: {:.2f} %'.format(accuracy))



# ====================== Data Loading =====================

# Load data and calculate average HSV and RGB value
train_day_hsv_rgb, train_night_hsv_rgb = data_initialize(mode='train')
val_day_hsv_rgb, val_night_hsv_rgb = data_initialize(mode='val')
test_day_hsv_rgb, test_night_hsv_rgb = data_initialize(mode='test')
len_train, len_val, len_test = len(train_day_hsv_rgb) + len(train_night_hsv_rgb), len(val_day_hsv_rgb) + len(val_night_hsv_rgb), len(test_day_hsv_rgb) + len(test_night_hsv_rgb) 

day_night_scatter(train_day_hsv_rgb, train_night_hsv_rgb, 'avg_V', 'avg_H', line_pos=30., loc_legend='lower left')



# ===============Imporve by H Channel Filter===============

''' Train Set (H Channel Filter)'''
# Calculate the maxmial accuracy
train_day_df_final, train_night_df_final = estimate_label_improve_H(train_day_hsv_rgb, train_night_hsv_rgb)
# Find the misclasified images
mis_train_day_df_final = get_misclassified_images(train_day_df_final)
mis_train_night_df_final = get_misclassified_images(train_night_df_final)
MISCLASSIFIED_train_final = pd.concat([mis_train_day_df_final, mis_train_night_df_final])

# Draw the scatter plot of misclassified images of train set 
scatter_plot_mis_images(MISCLASSIFIED_train_final, 'avg_V', 'avg_H', 'true_label', mode='train', loc_legend='upper left')



''' Validation Set (H Channel Filter)'''
# Calculate the maxmial accuracy
val_day_df_final, val_night_df_final =  estimate_label_improve_H(val_day_hsv_rgb, val_night_hsv_rgb)
# Find the misclasified images
mis_val_day_df_final = get_misclassified_images(val_day_df_final)
mis_val_night_df_final = get_misclassified_images(val_night_df_final)
MISCLASSIFIED_val_final = pd.concat([mis_val_day_df_final, mis_val_night_df_final])

# Draw the scatter plot of misclassified images of validation set
scatter_plot_mis_images(MISCLASSIFIED_val_final, 'avg_V', 'avg_H', 'true_label', mode='val', loc_legend='upper left')
visualize_mis_images(MISCLASSIFIED_val_final)



''' Test Set (H Channel Filter)'''
test_threshold = 81.0  # use the best threshold of the training set
test_day_df_final, test_night_df_final = estimate_label_improve_H(test_day_hsv_rgb, test_night_hsv_rgb, is_test=True, threshold=test_threshold)  
mis_test_day_df_final = get_misclassified_images(test_day_df_final)
mis_test_night_df_final = get_misclassified_images(test_night_df_final)
MISCLASSIFIED_test_final = pd.concat([mis_test_day_df_final, mis_test_night_df_final])

# Draw the scatter plot of misclassified images of test set
scatter_plot_mis_images(MISCLASSIFIED_test_final, 'avg_G', 'avg_H', 'true_label', mode='test', loc_legend='upper left')
visualize_mis_images(mis_test_night_df_final)




# ===============Imporve by RGB Channel Filter-===============

scatter_rgb(MISCLASSIFIED_train_final, 'avg_V',  'true_label', mode='train', loc_legend='upper left', title='MisClassified Images', is_mis=True)
hlines = [95, 130, 125]
scatter_rgb(train_day_df_final, 'avg_V',  'pred_label', mode='train', loc_legend='upper left', title='Day Images', hlines=hlines)
scatter_rgb(train_night_df_final, 'avg_V',  'pred_label', mode='train', loc_legend='upper left', title='Night Images', hlines=hlines)


'''  Train set (RGB Channel Filter) '''
# Get the improved train set through the RGB channel filter
train_day_df_final_G,  train_night_df_final_G = improve_RGB(train_day_df_final,  train_night_df_final)
# Find the misclassified images of train set
mis_train_day_df_final_G = get_misclassified_images(train_day_df_final_G)
mis_train_night_df_final_G = get_misclassified_images(train_night_df_final_G)
MISCLASSIFIED_train_final_G = pd.concat([mis_train_day_df_final_G, mis_train_night_df_final_G])

# Draw the scatter plot of misclassified images of train set
scatter_rgb(MISCLASSIFIED_train_final, 'avg_V',  'true_label', mode='train', loc_legend='upper left', title='MisClassified Images', is_mis=True)
scatter_rgb(MISCLASSIFIED_train_final_G, 'avg_V',  'true_label', mode='train', loc_legend='upper left', title='MisClassified Images After RGB', is_mis=True)


'''  Validation set (RGB Channel Filter) '''
# Get the improved validation set through the RGB channel filter
val_day_df_final_G,  val_night_df_final_G = improve_RGB(val_day_df_final,  val_night_df_final)
# Find the misclassified images of validation set
mis_val_day_df_final_G = get_misclassified_images(val_day_df_final_G)
mis_val_night_df_final_G = get_misclassified_images(val_night_df_final_G)
MISCLASSIFIED_val_final_G = pd.concat([mis_val_day_df_final_G, mis_val_night_df_final_G])

# Draw the scatter plot of misclassified images of validation set
scatter_rgb(MISCLASSIFIED_val_final, 'avg_V',  'true_label', mode='val', loc_legend='upper left', title='MisClassified Images', is_mis=True)
scatter_rgb(MISCLASSIFIED_val_final_G, 'avg_V',  'true_label', mode='val', loc_legend='upper left', title='MisClassified Images After RGB', is_mis=True)


'''  Test set (RGB Channel Filter) '''
scatter_rgb(MISCLASSIFIED_test_final, 'avg_V',  'true_label', mode='test', loc_legend='upper left', title='MisClassified Images', is_mis=True)
hlines = [95, 130, 125]
scatter_rgb(test_day_df_final, 'avg_V',  'pred_label', mode='test', loc_legend='upper left', title='Day Images', hlines=hlines)
scatter_rgb(test_night_df_final, 'avg_V',  'pred_label', mode='test', loc_legend='upper left', title='Night Images', hlines=hlines)

# Get the improved validation set through the RGB channel filter
test_day_df_final_G, test_night_df_final_G = improve_RGB(test_day_df_final, test_night_df_final)
# Find the misclassified images
mis_test_day_df_final_G = get_misclassified_images(test_day_df_final_G)
mis_test_night_df_final_G = get_misclassified_images(test_night_df_final_G)
MISCLASSIFIED_test_final_G = pd.concat([mis_test_day_df_final_G, mis_test_night_df_final_G])

# Draw the scatter plot of misclassified images of test set
scatter_rgb(MISCLASSIFIED_test_final, 'avg_V',  'true_label', mode='test', loc_legend='upper left', title='MisClassified Images', is_mis=True)
scatter_rgb(MISCLASSIFIED_test_final_G, 'avg_V',  'true_label', mode='test', loc_legend='upper left', title='MisClassified Images After RGB', is_mis=True)
visualize_mis_images(MISCLASSIFIED_test_final_G)

