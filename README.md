# Tumor detection using deep learning. 

Columbia University - Applied Deep Learning Project

Jinwoo Jung (jj2762@columbia.edu), Hyuk Joon Kwon (hk3084@columbia.edu)

## Project Overview

Notes for later:
<Mention how this project is based on a paper>

## Methodology

The data for the model needs to be gathered from the initial 21 different tissue images. We will iterate the entire image with a predefined stride. Starting with the top right corner patches of size (299\*299\*3) are collected by a moving window. A lot of the patches only contain only a small proportion of tissue so we will be getting rid of those patches using a given percentage. The center region of size (128*128) of each patch will determine it’s label. If there is at least one pixel of tumor detected in the center region, the label of that patch will be labeled as 0 (Tumor) otherwise (Normal). 

Moving window - stride
Tissue percentage filter
Creating patches
Deciding center region for label

## Objective

We had to decide on what makes a model good. Due to the extreme imbalance of Tumor and Normal patches a lot of the images will achieve over a 99 percent accuracy by simply predicting all the patches to be Normal. In real life it would not be such a big deal if the model predicted a patch to be Tumor when it was actually normal. A doctor can simply check themselves and dismiss the prediction. However, it would be problematic if a model predicted a patch to be normal when it actually contained cancer. Since the purpose of this model is to assist doctors and not replace them we decided that a high false negative rate is more problematic and so our objective is to try and achieve a high true positive rate. 

## Files and Folder

### 0.Initial_Data_Visualization.ipynb

This ipynb file describes a derivation of which biopsy and label images are used for train, valid, and test set. The raw dataset is composed of 22 biopsy and respective tumor mask images. Due to the difference in level dimensions, 1 of the images had to initially be taken out for consideration. 

With initial visual inspection of the images, we further decided to not use 6 of the images that rarely contain any tumors, as we later balance the number of patches of tissues that contain tumor to those without, from which those images would have a really small number of data for training a model.

 We further selected images that goes into train/val/test set such that 'good' biopsy and mask images are distributed more toward training set and test set (i.e. put rather 'poor' images in validation set since the set is used only for validation, and our goal is to have best prediction on the test set as much as we can). To make sure the model works on both images that have a lot of tumor and ones that don’t we made sure that both types of images were included in the test set. Image 78 was chosen because there were a lot of tumors in the image where image 64 and 91 were chosen because they had small but not tiny amounts of tumor in the image. 


Biopsy/mask img number in 
train set: [5, 16, 19, 23, 31, 84, 94, 101, 110]
valid set: [1, 75, 96]
test set: [64, 78, 91]

Image numbers 64, 78, 91 were used as the testing set. Although the zoom levels that were used were the same, the size of the images were all different. Image 64 was the biggest image and Image 78 was about half the size of image 64. Image 91 was around three times smaller compared to image 78. As Image 78 has the second largest amount of patches, we will use this as a reference to how much patches are used for prediction for each example.  


### 

## Models and Predictions

### Using Zoom level 3,4,5 - stride 150

Initially we started off by using three zoom levels 3,4 and 5. This was an initial model that used a striding window of 150. At this time we didn’t think of balancing the data between tumor and normal patches so there is a much greater number of normal patches compared to tumor patches. Image 78 produced 1548 patches, 36 patches on the x axis and 43 patches on the y axis. 

The predictions were not very good on all the different variants of models we trained. One thing in common was that the models underpredicted tumor patches thus the true positive rate was very low. This could be somewhat expected as our dataset is unbalanced. Also with a dimension of (36 * 43) there are actually not too many images in this zoom level which would have been another reason for the poor performance. 

We decided to move on to lower zoom levels such as zoom level 1,2,3,4. Also, due to the limitations of RAM and the fact that the paper mentioned that extra zoom levels were not as helpful we also created models of two different zoom levels instead of three. 

### (Base) Using zoom levels 2,3
   Prediction not good enough, so more vairations
###  Using zoom levels 2,3,4 - stride 200
### Using zoom levels 2,3,4 - stride 299
### Using zoom levels 1,2 

### Using Zoom levels 0,1
