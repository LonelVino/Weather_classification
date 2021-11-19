
import os
import numpy as np
import pandas as pd


from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
from tqdm import tqdm



def combine_into_df(data, path):
    ''' Combine the path and data into a dataframe
    
    Args:
        data (PIL.PngImagePlugin.PngImageFile): the pixel data of images
        path (string): the store path of each image
    
    Returns:
        pd.DataFrame: a DataFrame composed of image data and its path
    '''
    df = pd.DataFrame({'img': pd.Series(data), 'path': pd.Series(path)})
    return df


def load_images(weather_type, data_type):
    """
    Load images from the weather_type/data_type folder
    :param weather_type: fog or night or rain or snow
    :type weather_type: String
    :param data_type: train or val or test or train_ref or val_ref or test_ref
    :type data_type: String
    :return: dataframe, with columns = ['img', 'path']
    :rtype: pd.DataFrame
    """
    data = []
    data_paths = []
    counter = 0
    path = './images/' + weather_type + '/' + data_type + '/'

    # For each Gopro directory, for each image, store the image and its path in train and train_paths respectively
    for directory_name in os.listdir(path):
        gopro_path = path + directory_name
        for image_name in os.listdir(gopro_path):
            image_path = gopro_path + "/" + image_name
            image = Image.open(image_path)
            data.append(image)
            data_paths.append(image_path)

            # Counter to see progression
            counter += 1
            if counter%100 == 0:
                print(str(counter) + " " + data_type + " images loaded")
                
    df = combine_into_df(data, data_paths)
    
    return df



def get_accuracy(avg_b_day, avg_b_night, threshold=90.):
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




def find_max_accur(avg_b_day, avg_b_night, thresholds):
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
        accuracy.append(get_accuracy(avg_b_day, avg_b_night, threshold = thresh))
    accuracy_df = pd.DataFrame({
        'accuracy': pd.Series(accuracy, index=thresholds)
    })  # combine threshold and accuracy into a dataframe
    
    # Find the maximum value and corresponding index(threshold) of accuracy
    max_accuracy, max_threshold = accuracy_df.accuracy.max(), accuracy_df.accuracy.idxmax()
    
    # Print out the max accuracy and best threshold
    ratio_over_day = np.sum(np.array(avg_b_day) > max_threshold) / len(avg_b_day)
    ratio_under_night = 1 - np.sum(np.array(avg_b_night) > max_threshold) / len(avg_b_night)
    print("The ratio over {:.2f} in day_brightness is: {:.1f}%; \nThe ratio under {:.2f} in night_brightness is: {:.1f}%"          .format(max_threshold, ratio_over_day*100, max_threshold, ratio_under_night*100))
    
    # Visualize the distribution with the best threshold
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].hist(avg_b_day)
    ax[0].set_title('Day')
    ax[0].axvline(max_threshold, color='red')
    ax[1].hist(avg_b_night)
    ax[1].set_title('Night')
    ax[1].axvline(max_threshold, color='red')
    
    return accuracy_df, max_accuracy, max_threshold



def get_misclassified_images(df):
    ''' Get the list of misclassified images
    
    Args:
        df(pd.Dataframe): the original dataframe of images

    Returns:
        pd.Series of PIL.PngImagePlugin.PngImageFile: the misclassified images
    '''
    misclassified_images = df[df['pred_label'] != df['true_label']] 
    return misclassified_images



# Add HSV and RGB value with images
def avg_hsv_rgb(df_):
    ''' Calculate the average HSV and RGB of each image in a dataframe
    
    Args:
        df (pd.DataFrame): The dataframe to add the average HSV and RGB value
        
    Returns:
        3 lists: average Hue, average Saturation, average Value
    '''
    avg_Hs, avg_Ss, avg_Vs = [], [], []
    avg_Rs, avg_Gs, avg_Bs = [], [], []
    df = df_.copy()  # use copy() to avoid replacing the original dataframe
    print('[INFO] Calculating HSV and RGB Channel Value ......')
    for index in tqdm(range(len(df))):
        img = cv2.imread(df.iloc[index].path) # reading img 
        img = cv2.resize(img, (500, 500)) # resizing image
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # converting to hsv
        avg_H, avg_S, avg_V = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2]) # calculating average value per pixel of Value channel from HSV image
        avg_Hs.append(avg_H)
        avg_Ss.append(avg_S)
        avg_Vs.append(avg_V)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting to hsv
        avg_R, avg_G, avg_B = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2]) # calculating average value per pixel of Value channel from HSV image
        avg_Rs.append(avg_R)
        avg_Gs.append(avg_G)
        avg_Bs.append(avg_B)
    
    df['avg_H'], df['avg_S'], df['avg_V'] = avg_Hs, avg_Ss, avg_Vs   
    df['avg_R'], df['avg_G'], df['avg_B'] = avg_Rs, avg_Gs, avg_Bs  
    return df


def data_initialize(mode='val'):
    ''' Load the data set depending on the mode (train/valid/test), and attach average HSV and RGB
        Channel value with each image
        
    Args:
        mode (String): the type of dataset (train/ val/ test), default: val
    '''
    if mode == 'val':
        day_df, night_df = load_images('night', 'val_ref'), load_images('night', 'val')
    elif mode == 'train':
        day_df, night_df = load_images('night', 'train_ref'),  load_images('night', 'train')
    elif mode == 'test':
        day_df, night_df = load_images('night', 'test_ref'), load_images('night', 'test')

    # Calculate the average HSV and RGB value 
    day_df = avg_hsv_rgb(day_df)
    night_df = avg_hsv_rgb(night_df)
    
    return day_df, night_df



def day_night_scatter(df_day, df_night, x_name, y_name, line_pos, loc_legend='upper left'):
    ''' Scatter plot of Day Images and Night Images from the same dataset in a 1*2 subplots
    
    Args:
        df_day/day_night (pd.DataFrame): the day/night images with HSV and RGB value
        x_name (String): the name of X_axis, also the field of dataframe, such as `avg_H`, `avg_R`
        y_name (String): the name of Y_axis, also the field of dataframe, such as `avg_H`, `avg_R`
        label_name (String): the label of images, such as `true_label`, `pred_label`
        loc_legend (String): the location of the legend of the plot
    '''
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,2,1)
    ax1 = sns.scatterplot(x=x_name, y=y_name, data=df_day, legend='full')
    ax1.legend(loc=loc_legend)
    ax1.axhline(line_pos, color='red')
    ax2 = plt.subplot(1,2,2)
    ax2 = sns.scatterplot(x=x_name, y=y_name, data=df_night, legend='full')
    ax2.legend(loc=loc_legend)
    ax2.axhline(line_pos, color='red')
    plt.show()


def estimate_label_improve_H(day_df_, night_df_, is_test=False, **kwargs):
    ''' For train/valid, find the maximal accuracy and best threshold, 
        For test, use the specified threshold to calculate the accuracy
        For train/valid/test, add pred_label and true_label in DataFrame
        
    Args:
        day_df_/ night_df_ (pd.DataFrame): the dataframe of images
        is_test (Boolean): if True, test mode is on
    '''
    day_df, night_df = day_df_.copy(), night_df_.copy()
    
    if is_test == False:
        # First, find the best threshold to classify
        thresholds_70_90 = np.arange(70, 90, 0.1)
        accuracy_df, max_accuracy, max_threshold = find_max_accur(day_df.avg_V, night_df.avg_V, thresholds_70_90)
        print('The maximal accuracy is: {:.1f}%, where threshold is {:.1f}'.format(max_accuracy*100, max_threshold))
    else:
        max_threshold = kwargs['threshold']
        test_accuracy = get_accuracy(day_df.avg_V, night_df.avg_V, max_threshold)
        print('The accuracy of test set is: {:.1f}%, where threshold is {:.1f}'.format(test_accuracy*100, max_threshold))
    
    # Second, classify images with H Channel Improvement and add labels
    day_df.loc[(day_df['avg_V'] <= max_threshold) |                ((day_df['avg_V'] > max_threshold) & (day_df['avg_H'] <= 35.)), 'pred_label'] = 'Night' 
    day_df.fillna({'pred_label': 'Day'}, inplace=True)
    night_df.loc[(night_df['avg_V'] <= max_threshold) |                  ((night_df['avg_V'] > max_threshold) & (night_df['avg_H'] <= 35.)), 'pred_label'] = 'Night' 
    night_df.fillna({'pred_label': 'Day'}, inplace=True) 
    
    day_df['true_label'] = 'Day'
    night_df['true_label'] = 'Night'
    
    return day_df, night_df


def improve_RGB(val_day_df_, val_night_df_):
    val_day_df, val_night_df = val_day_df_.copy(), val_night_df_.copy()
    
    val_day_df.loc[(val_day_df['pred_label'] == 'Night') & (val_day_df['avg_R'] >= 95.), 'pred_label'] = 'Day' 
    val_night_df.loc[(val_night_df['pred_label'] == 'Night') & (val_night_df['avg_R'] >= 95.), 'pred_label'] = 'Day'
    
    val_day_df.loc[(val_day_df['pred_label'] == 'Day') & (val_day_df['avg_G'] >= 130.), 'pred_label'] = 'Night' 
    val_night_df.loc[(val_night_df['pred_label'] == 'Day') & (val_night_df['avg_G'] >= 130.), 'pred_label'] = 'Night'
    
    val_day_df.loc[(val_day_df['pred_label'] == 'Day') & (val_day_df['avg_G'] >= 130.), 'pred_label'] = 'Night' 
    val_night_df.loc[(val_night_df['pred_label'] == 'Day') & (val_night_df['avg_G'] >= 130.), 'pred_label'] = 'Night'
    
    return val_day_df, val_night_df 
    
