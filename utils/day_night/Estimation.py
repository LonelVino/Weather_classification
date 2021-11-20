import numpy as np
import pandas as pd
from tqdm import tqdm

from .Visualization import *


def get_accuracy(avg_b_day=[], avg_b_night=[], threshold=90.):
    ''' Calculate the accuracy of brightness of day and night
    
    Args:
        avg_b_day (list of float64): the average brightness of day (train set /val set /test set)
        avg_b_day (list of float64): the average brightness of night (train set /val set /test set)
        threshold (float64): the threshold brightness to classify day/night   
        
    Returns:
        float64: the accuracy of classification of day/night
    '''
    corrects = 0 # tracks running correct values
    total = len(avg_b_day) + len(avg_b_night) # total number of images in set
    
    corrects = sum(np.array(avg_b_day) > threshold) + sum(np.array(avg_b_night) < threshold)
    accuracy = corrects/total # calculating percentage of correctly classified images
    return accuracy


def get_misclassified_images(df):
    ''' Get the list of misclassified images
    
    Args:
        df(pd.Dataframe): the original dataframe of images

    Returns:
        pd.Series of PIL.PngImagePlugin.PngImageFile: the misclassified images
    '''
    misclassified_images = df[df['pred_label'] != df['true_label']] 
    return misclassified_images


    
def calc_mis_accur(MISCLASSIFIED_hsv_rgb, length):
    ''' Calculate the accuracy according to the misclassified images
    Args:
        MISCLASSIFIED_hsv_rgb (pd.DataFrame): the misclassified images with HSV and RGB value

    Returns:
        float64: the accuracy according to the misclassified images
    '''
    accuracy = (1-len(MISCLASSIFIED_hsv_rgb)/ length)*100
    return accuracy



def find_best_V(avg_b_day, avg_b_night, thresholds):
    ''' Traverse the thresolds to find the maximal accuracy and corresponding index(threshold), according to the avearge brightness
    
    Args:
        avg_b_day (list of float64): the average brightness of day (train set /val set /test set)
        avg_b_day (list of float64): the average brightness of night (train set /val set /test set)
        thresholds (list of float64): the brightness thresholds to classify day/night   
        
    Returns:
        pd.DataFrame: the table of (threshold, accuracy)
        float64: The maximal validation accuracy and corresponding threshold
    '''
    accuracy = []
    for thresh in thresholds:
        accuracy.append(get_accuracy(avg_b_day=avg_b_day, avg_b_night=avg_b_night, threshold = thresh))
    accuracy_df = pd.DataFrame({
        'accuracy': pd.Series(accuracy, index=thresholds)
    })  # combine threshold and accuracy into a dataframe
    
    # Find the maximum value and corresponding index(threshold) of accuracy
    max_accuracy, max_threshold = accuracy_df.accuracy.max(), accuracy_df.accuracy.idxmax()
    
    # Print out the max accuracy and best threshold
    ratio_over_day = np.sum(np.array(avg_b_day) > max_threshold) / len(avg_b_day)
    ratio_under_night = 1 - np.sum(np.array(avg_b_night) > max_threshold) / len(avg_b_night)
    print('The maximal accuracy is: {:.2f}% w.r.t. Value Channel. \nThe best Value threshold  is {:.2f}.\n{:s}'.\
        format(max_accuracy*100, max_threshold, '+ '*25))
    
    return max_accuracy, max_threshold


def find_best_H(df_day, df_night, Hs, best_V_threshold, mode):
    ''' Traverse the thresolds to find the maximal accuracy and corresponding index(threshold), according to the avearge brightness
    
    Args:
        df_day (pd.DataFrame): the dataframe of day (train set /val set /test set)
        df_night (pd.DataFrame): the dataframe of night (train set /val set /test set)
        Hs (list of float64): the HUE Values to classify Correct the misclassification of day/night   
        mode (String): train/ valid mode
        
    Returns:
    for i in range(num_type):
        for j in range(3):
        float64: The maximal improved accuracy and corresponding Hue Value threshold
    '''
    accuracy = []
    print('[INFO] Traversing to find the Best Hue Channel Threshold......')
    for H in tqdm(Hs):
        day_df, night_df = df_day.copy(), df_night.copy()
        day_df.loc[(day_df['avg_V'] <= best_V_threshold) | \
            ((day_df['avg_V'] > best_V_threshold) & (day_df['avg_H'] <= H)), 'pred_label'] = 'Night' 
        day_df.fillna({'pred_label': 'Day'}, inplace=True)
        night_df.loc[(night_df['avg_V'] <= best_V_threshold) | \
            ((night_df['avg_V'] > best_V_threshold) & (night_df['avg_H'] <= H)), 'pred_label'] = 'Night' 
        night_df.fillna({'pred_label': 'Day'}, inplace=True) 

        day_df['true_label'] = 'Day'
        night_df['true_label'] = 'Night'
        
        mis_day_df = get_misclassified_images(day_df)
        mis_night_df = get_misclassified_images(night_df)
        MISCLASSIFIED = pd.concat([mis_day_df, mis_night_df])
    
        accuracy.append(calc_mis_accur(MISCLASSIFIED, len(day_df)+len(night_df)))
    
    accuracy_df = pd.DataFrame({
        'accuracy': pd.Series(accuracy, index=Hs)
    })  # combine threshold and accuracy into a dataframe
    
    # Find the maximum value and corresponding index(threshold) of accuracy
    max_accuracy, best_H_threshold = accuracy_df.accuracy.max(), accuracy_df.accuracy.idxmax()
    
    # Print out the max accuracy and best threshold
    print("{:s}\nThe maximal improved accuracy is {:.2f}% w.r.t Hue Channel in {:s} mode.\nThe best Hue value threshold is {:.2f}.\n{:s}".format('+ '*25, max_accuracy, mode, best_H_threshold, '+ '*25))
    
    # Visualize the distribution with the best threshold
    day_night_scatter(df_day, df_night, 'avg_V', 'avg_H', best_H_threshold, mode, loc_legend='upper left')
    
    return max_accuracy, best_H_threshold



def estimate_label_improve_H(day_df_, night_df_, mode, **kwargs):
    ''' For train/valid, find the maximal accuracy and best threshold, 
        For test, use the specified threshold to calculate the accuracy
        For train/valid/test, add pred_label and true_label in DataFrame
        
    Args:
        day_df_/ night_df_ (pd.DataFrame): the dataframe of images
        is_test (Boolean): if True, test mode is on
    '''
    day_df, night_df = day_df_.copy(), night_df_.copy()
    
    if mode != 'test':
        # First, find the best threshold to classify
        Vs_70_90 = np.arange(70, 90, 0.1)
        max_V_accuracy, best_V_threshold = find_best_V(day_df.avg_V, night_df.avg_V, Vs_70_90)
        # Second, traverse to find the best H value
        Hs_20_50 = np.arange(20, 50, 1)
        max_H_accuracy, best_H_threshold = find_best_H(day_df, night_df, Hs_20_50, best_V_threshold, mode=mode)
    else:
        best_V_threshold, best_H_threshold = kwargs['V_threshold'], kwargs['H_threshold']
        max_V_accuracy = get_accuracy(avg_b_day=day_df.avg_V, avg_b_night=night_df.avg_V, threshold=best_V_threshold)
    
    day_df.loc[(day_df['avg_V'] <= best_V_threshold) | \
        ((day_df['avg_V'] > best_V_threshold) & (day_df['avg_H'] <= best_H_threshold)), 'pred_label'] = 'Night' 
    day_df.fillna({'pred_label': 'Day'}, inplace=True)
    night_df.loc[(night_df['avg_V'] <= best_V_threshold) | \
        ((night_df['avg_V'] > best_V_threshold) & (night_df['avg_H'] <= best_H_threshold)), 'pred_label'] = 'Night' 
    night_df.fillna({'pred_label': 'Day'}, inplace=True) 
    
    day_df['true_label'] = 'Day'
    night_df['true_label'] = 'Night'
    
    if mode == 'test':
        mis_test_day_df = get_misclassified_images(day_df)
        mis_test_night_df = get_misclassified_images(night_df)
        MISCLASSIFIED_test_final = pd.concat([mis_test_day_df, mis_test_night_df])
        max_H_accuracy = calc_mis_accur(MISCLASSIFIED_test_final, len(day_df)+len(night_df))
        print('{:s}\nThe accuracy of test set is: {:.2f}% w.r.t. Value Channel in {:s} mode.\nThe Value threshold is {:.2f}\n{:s}'.format( '+ '*25, max_V_accuracy*100, mode, best_V_threshold, '+ '*25))
        print('{:s}\nThe accuracy of test set is: {:.2f}% w.r.t. Hue Channel in {:s} mode.\nThe Hue threshold  is {:.2f}\n{:s}'.format( '+ '*25, max_H_accuracy, mode, best_H_threshold, '+ '*25))

    
    return day_df, night_df, max_V_accuracy, best_V_threshold, max_H_accuracy, best_H_threshold


def find_best_RGB(df_day, df_night, Rs, Gs, mode):
    ''' Traverse the thresolds to find the maximal accuracy and corresponding index(threshold), according to the avearge brightness
    
    Args:
        df_day (pd.DataFrame): the dataframe of day (train set /val set /test set)
        df_night (pd.DataFrame): the dataframe of night (train set /val set /test set)
        Hs (list of float64): the HUE Values to classify Correct the misclassification of day/night   
        mode (String): train/ valid mode
        
    Returns:
        float64: The maximal improved accuracy and corresponding Hue Value threshold
    '''
    max_accuracy, best_R, best_G = 0., 0., 0.
    print('[INFO] Traversing to find the Best R and G Channel Threshold......')
    for R in tqdm(Rs):
        for G in Gs:
            day_df, night_df = df_day.copy(), df_night.copy()
            
            day_df.loc[(day_df['pred_label'] == 'Night') & (day_df['avg_R'] >= R), 'pred_label'] = 'Day' 
            night_df.loc[(night_df['pred_label'] == 'Night') & (night_df['avg_R'] >= R), 'pred_label'] = 'Day'
            day_df.loc[(day_df['pred_label'] == 'Day') & (day_df['avg_G'] >= G), 'pred_label'] = 'Night' 
            night_df.loc[(night_df['pred_label'] == 'Day') & (night_df['avg_G'] >= G), 'pred_label'] = 'Night'

            mis_day_df = get_misclassified_images(day_df)
            mis_night_df = get_misclassified_images(night_df)
            MISCLASSIFIED = pd.concat([mis_day_df, mis_night_df])
            accuracy = calc_mis_accur(MISCLASSIFIED, len(day_df)+len(night_df))
            
            if accuracy > max_accuracy:
                max_accuracy  = accuracy
                best_R, best_G = R, G
                
    
    # Print out the max accuracy and best threshold
    print("{:s}\nThe maximal improved accuracy is {:.2f}% w.r.t RGB Channel in {:s} mode.\nThe best R value threshold is {:.2f}.\nThe best G value threshold is {:.2f}.\n{:s}".\
        format( '+ '*25, max_accuracy, mode, best_R, best_G, '+ '*25))

    return max_accuracy, best_R, best_G


def improve_RGB(day_df_, night_df_, mode, **kwargs):
    ''' For train/valid, find the maximal accuracy and best threshold, 
        For test, use the specified threshold to calculate the accuracy
        For train/valid/test, add pred_label and true_label in DataFrame
        
    Args:
        day_df_/ night_df_ (pd.DataFrame): the dataframe of images
        is_test (Boolean): if True, test mode is on
    '''
    day_df, night_df = day_df_.copy(), night_df_.copy()
    
    if mode != 'test':
        # First, find the best threshold to classify
        Rs_90_110, Gs_110_140 = np.arange(90, 110, 1), np.arange(110, 140, 1)
        max_RGB_accuracy, R_threshold, G_threshold = find_best_RGB(day_df, night_df, Rs_90_110, Gs_110_140, mode)
    else:
        R_threshold, G_threshold = kwargs['R_threshold'], kwargs['G_threshold']
    
    day_df.loc[(day_df['pred_label'] == 'Night') & (day_df['avg_R'] >= R_threshold), 'pred_label'] = 'Day' 
    night_df.loc[(night_df['pred_label'] == 'Night') & (night_df['avg_R'] >= R_threshold), 'pred_label'] = 'Day'
    day_df.loc[(day_df['pred_label'] == 'Day') & (day_df['avg_G'] >= G_threshold), 'pred_label'] = 'Night' 
    night_df.loc[(night_df['pred_label'] == 'Day') & (night_df['avg_G'] >= G_threshold), 'pred_label'] = 'Night'
    
    if mode == 'test':
        mis_test_day_df = get_misclassified_images(day_df)
        mis_test_night_df = get_misclassified_images(night_df)
        MISCLASSIFIED_test_final = pd.concat([mis_test_day_df, mis_test_night_df])
        max_RGB_accuracy = calc_mis_accur(MISCLASSIFIED_test_final, len(day_df)+len(night_df))
        print('{:s}\nThe accuracy of test set is: {:.2f}% w.r.t. RGB Channel in {:s} mode.\nThe R threshold  is {:.2f}\n The G threshold is {:.2f}.\n {:s}'.format( '+ '*25, max_RGB_accuracy, mode, R_threshold, G_threshold, '+ '*25))

    
    return day_df, night_df, max_RGB_accuracy, R_threshold, G_threshold
