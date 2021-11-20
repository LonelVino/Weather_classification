import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm


'''
Define the data loading function
    - `combine_into_df()`: Combine the path and data into a dataframe
    - `load_image()`: Load images from the weather_type/data_type folder
    - `data_initialize()`: Load the data set depending on the mode (train/valid/test), 
                    and attach average HSV and RGB Channel value with each image
    - `avg_hsv_rgb()`:  Calculate the average HSV and RGB of each image in a dataframe
'''

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

