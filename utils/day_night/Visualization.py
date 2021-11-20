import math
import random

import matplotlib.pyplot as plt
import seaborn as sns

'''
# Define the Visualization functions of misclassified images
    - `day_night_scatter()`: Scatter plot of Day Images and Night Images from the same dataset in a 1*2 subplots, and show the threshold with lines
    - `scatter_plot_mis_images()`: show the misclassified images in a scatter plot.
    - `visualize_mis_images()`: Visualize misclassified example(s), show the true label / brigtness / predicted label
    - `scatters_rgb`: Scatter plot of images in (1 * 3) subplots
                    X axis is a Channel Value selected from HSV and RGB,
                    Y axis is R Channel, G Channel, B Channel respectively in each subplot
'''

def day_night_scatter(df_day, df_night, x_name, y_name, line_pos, mode, loc_legend='upper left'):
    ''' Scatter plot of Day Images and Night Images from the same dataset in a 1*2 subplots, and show the threshold with lines
    
    Args:
        df_day/day_night (pd.DataFrame): the day/night images with HSV and RGB value
        x_name (String): the name of X_axis, also the field of dataframe, such as `avg_H`, `avg_R`
        y_name (String): the name of Y_axis, also the field of dataframe, such as `avg_H`, `avg_R`
        label_name (String): the label of images, such as `true_label`, `pred_label`
        loc_legend (String): the location of the legend of the plot
    '''
    fig = plt.figure(figsize=(12,6))
    fig.suptitle(' Threshold of Day and Night Images (Mode: {:s})'.format(mode), fontsize=15)

    ax1 = plt.subplot(1,2,1)
    ax1 = sns.scatterplot(x=x_name, y=y_name, data=df_day, legend='full')
    ax1.set_title('Day Images')
    ax1.axhline(line_pos, color='red')
    ax1.annotate(str(line_pos), (line_pos+5, line_pos+5), color='r', weight='bold',  size=10)
    ax2 = plt.subplot(1,2,2)
    ax2 = sns.scatterplot(x=x_name, y=y_name, data=df_night, legend='full')
    ax2.set_title('Night Images')
    ax2.axhline(line_pos, color='red')
    plt.show()


def scatter_plot_mis_images(MISCLASSIFIED_hsv_rgb, x_name, y_name, label_name, accuracy ,mode, loc_legend='lower left'):
    ''' Visualize misclassified example(s) with scatter plot, and print out the accuracy of classification
    
    Args:
        MISCLASSIFIED_hsv_rgb (pd.DataFrame): the misclassified images with HSV and RGB value
        x_name (String): the name of X_axis, also the field of dataframe, such as `avg_H`, `avg_R`
        y_name (String): the name of Y_axis, also the field of dataframe, such as `avg_H`, `avg_R`
        label_name (String): the label of images, such as `true_label`, `pred_label`
        mode (String): 3 choices -- train / validation / test mode
        loc_legend (String): the location of the legend of the plot
    ''' 
    ax = sns.scatterplot(x=x_name, y=y_name, hue=label_name, data=MISCLASSIFIED_hsv_rgb, legend='full')
    ax.legend(loc=loc_legend)
    ax.set_title('MISCLASSIFED Images ({:s}) [{:s} -- {:s}]'.format(x_name, y_name, mode), fontsize=15)
    
    ann_text = 'Accuracy: ' + str(accuracy) + '%' + '\n' + 'Number of Misclassified Images: ' + str(len(MISCLASSIFIED_hsv_rgb))
    ax.text(0.1, 0.1, ann_text, color='r', weight='bold',  size=10,  transform=ax.transAxes)
    
    plt.show()
        
    
def visualize_mis_images(mis_images):
    ''' Visualize misclassified example(s), show the true label / brigtness / predicted label
    
    Args:
        mis_images (pd.DataFrame): the misclassified images
    '''
    len_images = len(mis_images) if len(mis_images)<=25 else 25
    num = math.ceil(math.sqrt(len_images))
    idxs = random.sample(range(0, len(mis_images)), len_images)
    
    fig = plt.figure(figsize=(num**2,num**2)) if num > 3 else  plt.figure(figsize=((num+1)**2,(num+1)**2))
    plt.title("Misclassified images (True Label - Brightness - Predicted Label)",  fontsize=16)
    for count, index in enumerate(idxs):
        ax = fig.add_subplot(num, num, count + 1, xticks=[], yticks=[])
        image = mis_images.iloc[index].img
        label_true = mis_images.iloc[index].true_label
        label_pred = mis_images.iloc[index].pred_label
        bright = mis_images.iloc[index]['avg_b'] if 'avg_b' in mis_images else  mis_images.iloc[index]['avg_V'] 
        ax.imshow(image)
        ax.set_title("{} {:0.0f} {}".format(label_true, bright, label_pred))
    
        if count==len_images-1:
            plt.show()
            break    


def scatters_rgb(df, x_name, label_name, mode, num_type, loc_legend, title, hlines=None, is_mis=False):
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
    '''
    
    rgb = ['avg_R', 'avg_G', 'avg_B']
    f = plt.figure(figsize=(20, num_type*5))
    f.suptitle(' RGB Scatter Plot [Red / Green / Black] (Mode: {:s})'.format(mode), fontsize=15)   
            
    for i in range(num_type):
        for j in range(3):
            idx = (i*3) + (j+1) # first round: 1, 2, 3; second round: 4, 5, 6...
            ax = f.add_subplot(num_type, 3 ,idx)
            ax = sns.scatterplot(x=x_name[i], y=rgb[j], hue=label_name[i], data=df[i], legend='full')
            ax.legend(loc=loc_legend[i])
            if i != 0 and hlines != None:
                ax.axhline(hlines[j], color='red')
            ax.set_title('[{:s}] {:s} -- {:s}'.format(title[i], x_name[i], rgb[i])) 
            
    plt.show()
    
    
    