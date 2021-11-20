#!/usr/bin/env python
# coding: utf-8

from utils.day_night.DataLoader import *
from utils.day_night.Visualization import *
from utils.day_night.Estimation import *

import pandas as pd


# ====================== Data Loading =====================

# Load data and calculate average HSV and RGB value
train_day_hsv_rgb, train_night_hsv_rgb = data_initialize(mode='train')
val_day_hsv_rgb, val_night_hsv_rgb = data_initialize(mode='val')
test_day_hsv_rgb, test_night_hsv_rgb = data_initialize(mode='test')
len_train, len_val, len_test = len(train_day_hsv_rgb) + len(train_night_hsv_rgb), len(val_day_hsv_rgb) + len(val_night_hsv_rgb), len(test_day_hsv_rgb) + len(test_night_hsv_rgb) 


# ===============Imporve by H Channel Filter===============

''' Train Set (H Channel Filter)'''
print('\n', '='*20, 'Try to find the best Value and Hue Channel Threshold of Train set', '='*20)
# find the best Value and Hue Channel Threshold
train_day_df_final, train_night_df_final, train_max_V_accuracy, train_best_V_threshold,\
    train_max_H_accuracy, train_best_H_threshold = estimate_label_improve_H(train_day_hsv_rgb, train_night_hsv_rgb, mode='train')
# Find the misclasified images
mis_train_day_df_final = get_misclassified_images(train_day_df_final)
mis_train_night_df_final = get_misclassified_images(train_night_df_final)
MISCLASSIFIED_train_final = pd.concat([mis_train_day_df_final, mis_train_night_df_final])

# Draw the scatter plot of misclassified images of train set 
scatter_plot_mis_images(MISCLASSIFIED_train_final, 'avg_V', 'avg_H', 'true_label', train_max_H_accuracy, mode='train', loc_legend='upper left')
visualize_mis_images(MISCLASSIFIED_train_final)



''' Validation Set (H Channel Filter)'''
print('\n', '='*20, 'Try to find the best Value and Hue Channel Threshold of Validation set', '='*20)
# find the best Value and Hue Channel Threshold 
val_day_df_final, val_night_df_final, val_max_V_accuracy, val_best_V_threshold, \
    val_max_H_accuracy, val_best_H_threshold = estimate_label_improve_H(val_day_hsv_rgb, val_night_hsv_rgb, mode='val')
# Find the misclasified images
mis_val_day_df_final = get_misclassified_images(val_day_df_final)
mis_val_night_df_final = get_misclassified_images(val_night_df_final)
MISCLASSIFIED_val_final = pd.concat([mis_val_day_df_final, mis_val_night_df_final])

# Draw the scatter plot of misclassified images of validation set
scatter_plot_mis_images(MISCLASSIFIED_val_final, 'avg_V', 'avg_H', 'true_label', val_max_H_accuracy, mode='val', loc_legend='upper left')
visualize_mis_images(MISCLASSIFIED_val_final)



''' Test Set (H Channel Filter)'''
print('\n', '='*20, 'Use the best H an V threshold of train set to find the misclasified images', '='*20)
# Use the best H an V threshold of train set to find the misclasified images
test_day_df_final, test_night_df_final, test_max_V_accuracy, test_best_V_threshold, test_max_H_accuracy, test_best_H_threshold \
    = estimate_label_improve_H(test_day_hsv_rgb, test_night_hsv_rgb, mode='test', V_threshold=train_best_V_threshold, H_threshold=train_best_H_threshold)  
mis_test_day_df_final = get_misclassified_images(test_day_df_final)
mis_test_night_df_final = get_misclassified_images(test_night_df_final)
MISCLASSIFIED_test_final = pd.concat([mis_test_day_df_final, mis_test_night_df_final])

# Draw the scatter plot of misclassified images of test set
scatter_plot_mis_images(MISCLASSIFIED_test_final, 'avg_G', 'avg_H', 'true_label', test_max_H_accuracy, mode='test', loc_legend='upper left')
visualize_mis_images(MISCLASSIFIED_test_final)






# ===============Imporve by RGB Channel Filter-===============

'''  Train set (RGB Channel Filter) '''
print('\n', '='*20, 'Try to find the best Red and Grren Channel Threshold of Train set', '='*20)
# Get the improved train set by filtering the RGB channel
train_day_df_final_RGB, train_night_df_final_RGB, max_RGB_accuracy, R_threshold, G_threshold = improve_RGB(train_day_df_final,  train_night_df_final, 'train')

# Find the misclassified images of train set
mis_train_day_df_final_RGB = get_misclassified_images(train_day_df_final_RGB)
mis_train_night_df_final_RGB = get_misclassified_images(train_night_df_final_RGB)
MISCLASSIFIED_train_final_RGB = pd.concat([mis_train_day_df_final_RGB, mis_train_night_df_final_RGB])

# Draw the 2 scatter plots of misclassified images of train set
dfs_mis_train = [MISCLASSIFIED_train_final, MISCLASSIFIED_train_final_RGB]
x_names_mis, label_names_mis = ['avg_V']*2, ['true_label', 'true_label']
mode_mis_train, num_type_mis = 'train', 2 # 3 kinds of images: misclassified / day/ night
loc_legends_mis, titles_mis = ['upper left']*2, ['MisClassified ', 'MisClassified after RGB'] 

scatters_rgb(dfs_mis_train, x_names_mis, label_names_mis, mode_mis_train, num_type_mis, loc_legends_mis, titles_mis)
accuracy_RGB_train = [calc_mis_accur(dfs_mis_train[i], len_train) for i in range(num_type_mis)]
print('The accuracy accordinig to the {:s} {:s} Images: {:.2f} %'.format(titles_mis[0], mode_mis_train, accuracy_RGB_train[0]))
print('The accuracy accordinig to the {:s} {:s} Images: {:.2f} %'.format(titles_mis[1], mode_mis_train, accuracy_RGB_train[1]))



'''  Validation set (RGB Channel Filter) '''
print('\n', '='*20, 'Try to find the best Red and Grren Channel Threshold of Validation set', '='*20)
# Get the improved validation set by filtering the RGB channel
val_day_df_final_RGB, val_night_df_final_RGB, max_RGB_accuracy_val, R_threshold_val, G_threshold_val = improve_RGB(val_day_df_final, val_night_df_final, 'val')

# Find the misclassified images of validation set
mis_val_day_df_final_RGB = get_misclassified_images(val_day_df_final_RGB)
mis_val_night_df_final_RGB = get_misclassified_images(val_night_df_final_RGB)
MISCLASSIFIED_val_final_RGB = pd.concat([mis_val_day_df_final_RGB, mis_val_night_df_final_RGB])

# Draw the scatter plot of misclassified images of validation set
dfs_mis_val = [MISCLASSIFIED_val_final, MISCLASSIFIED_val_final_RGB]
mode_mis_val = 'val'
scatters_rgb(dfs_mis_val, x_names_mis, label_names_mis, mode_mis_val, num_type_mis, loc_legends_mis, titles_mis)
accuracy_RGB_val = [calc_mis_accur(dfs_mis_val[i], len_val) for i in range(num_type_mis)]
print('The accuracy accordinig to the {:s} {:s} Images: {:.2f} %'.format(titles_mis[0], mode_mis_val, accuracy_RGB_val[0]))
print('The accuracy accordinig to the {:s} {:s} Images: {:.2f} %'.format(titles_mis[1], mode_mis_val, accuracy_RGB_val[1]))



'''  Test set (RGB Channel Filter) '''
print('\n', '='*20, 'Try to find the best Red and Grren Channel Threshold of Test set', '='*20)
# Get the improved test set through the RGB channel filter
test_day_df_final_RGB, test_night_df_final_RGB, max_RGB_accuracy_test, R_threshold_test, G_threshold_test \
    = improve_RGB(test_day_df_final, test_night_df_final, 'test', R_threshold=R_threshold, G_threshold=G_threshold)
# Find the misclassified images
mis_test_day_df_final_RGB = get_misclassified_images(test_day_df_final_RGB)
mis_test_night_df_final_RGB = get_misclassified_images(test_night_df_final_RGB)
MISCLASSIFIED_test_final_RGB = pd.concat([mis_test_day_df_final_RGB, mis_test_night_df_final_RGB])


# Draw the scatter plot of misclassified images of test set
dfs_mis_test = [MISCLASSIFIED_test_final, MISCLASSIFIED_test_final_RGB]
mode_mis_test = 'test'
scatters_rgb(dfs_mis_test, x_names_mis, label_names_mis, mode_mis_test, num_type_mis, loc_legends_mis, titles_mis, is_mis=True)
accuracy_RGB_test = [calc_mis_accur(dfs_mis_test[i], len_test) for i in range(num_type_mis)]
print('The accuracy accordinig to the {:s} {:s} Images: {:.2f} %'.format(titles_mis[0], mode_mis_test, accuracy_RGB_test[0]))
print('The accuracy accordinig to the {:s} {:s} Images: {:.2f} %'.format(titles_mis[1], mode_mis_test, accuracy_RGB_test[1]))
# Draw the scatter plot of misclassified images of test set
visualize_mis_images(MISCLASSIFIED_test_final_RGB)



''' Visualize the images of train set with RGB threshold by 9 scatter plots'''

dfs_train = [MISCLASSIFIED_train_final, train_day_df_final, train_night_df_final]
x_names, label_names = ['avg_V']*3, ['true_label', 'pred_label', 'pred_label']
mode_train, num_type = 'train', 3 # 3 kinds of images: misclassified / day/ night
loc_legends, titles = ['upper left']*3, ['MisClassified', 'Day', 'Night'] 
hlines = [R_threshold, G_threshold, 125] 
scatters_rgb(dfs_train, x_names, label_names, mode_train, num_type, loc_legends, titles, hlines)

# Visualize the images of test set with RGB threshold by 9 scatter plots
dfs_test = [MISCLASSIFIED_test_final, test_day_df_final, test_night_df_final]
mode_test = 'test'
scatters_rgb(dfs_test, x_names, label_names, mode_test, num_type, loc_legends, titles, hlines)