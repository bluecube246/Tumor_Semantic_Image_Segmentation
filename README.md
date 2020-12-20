# Tumor detection using deep learning. 

Columbia University - Applied Deep Learning Project

Jinwoo Jung (jj2762@columbia.edu), Hyuk Joon Kwon (hk3084@columbia.edu)


## Project Overview
This project is to build a tool that produces heatmaps that identify regions of tumors in tissues of biopsy images that works as an automatic second opinion for pathologists. The provided sample datasets are originally from CAMELYON16  Challenge’s 400 WSI( whole slide images), which are collected independently from two medical centers in the Netherlands.

## Objective
It is important to decide on what makes a model good. Due to the extreme imbalance of Tumor and Normal patches such that there are much more patch images that are classified as having tumor than not, a lot of the images will achieve over a 99 percent accuracy by simply predicting all the patches to be Normal. In real life it would not be such a big deal if the model predicted a patch to be Tumor when it was actually normal. A doctor can simply check themselves and dismiss the prediction. However, it would be problematic if a model predicted a patch to be normal when it actually contained cancer. Since the purpose of this model is to assist doctors and not replace them, our group decided that a high false negative rate is more problematic. Thus, our objective is to try and achieve a high true positive rate (equally low false negative rate) at the same time trying to achieve high AUC accuracy.

## Methodology 
### Sliding window approach
For each biopsy and mask slides in the training set, we first tried to create patches. We decided to adopt a **sliding-window** method such that with a given hyper-parameter of **stride** size (ex. 299),  biopsy slide,  and mask slide of a certain image number, we create a list of patches from top to bottom of image, and from left to right starting from top (i.e. starts from top-left, ends at bottom-left side of a slide).

### Minimum tissue percentage 
However, since we could not store all the patches in this basic method in RAM provided even in Google Colab Pro version when we look in to relatively lower slide (i.e. as the slide’s level decreases, more resolution, thus bigger image size), we adopted to apply filtering process of taking only patches that have higher tissue percentage than a predefined **minimum percentage of tissue** in a biopsy image (ex. 50%).  

### Balancing data
Since the distribution of collected patches of biopsy and mask images, even after the above filtering process, results in imbalance data such that there are much larger number of patches that are classified as non-tumor, we further adopted downsampling(undersampling) to balance number of patches that are classified as having tumor to those without having tumor.


### Multiple zoom levels and classifying sub-patch
Of course, since we are utilizing multiple zoom levels (we used both 2 levels and 3 levels), and since each image and each level’s dimension is different, we applied transformation of coordinates appropriately to locate to the same center position throughout multiple zoom levels. Furthermore, we performed sliding with the lowest level (i.e. if the stride is 200 the sliding of 200 will be performed with the lowest level). The sliding will continue until the patch has reached the end of the highest level (i.e. most zoomed out version) coordinates. This is to prevent the highest level patch from moving outside the legal coordinates. As instructed in class, for each patch, we looked at the center sub-patch region of 128x 128, and labeled/classified each patch to have tumor or not by identifying if there is at least a single pixel of tumor detected in the sub-patch region. Thus, the label of that patch will be labeled 1 (Tumor) otherwise 0 (Normal). 

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

Note: When analyzing zoom levels 0,1, we decided to further decrease samples used to be able to run within the Colab Pro’s RAM limit. Thus, only for this specific case, we used image number of 75, 94, 101, 110 for train set, 78 for validation set, and 91 for test set.

## Models and Predictions

### Models
There are 7 different kinds of models that we used to train and to produce heatmaps.
The vanilla model (model version a) we adopted is by concatenating two InceptionV3 models and adding a dense layer on top of it and another dense layer to classify its label. We then tried to adopt both fine tuning each Inception V3 model (i.e. fine tune at 100 layer from original 311 layers), and applying data augmentation. This gave 4 different models from version a to d. Furthermore, since these variations did not seem to give good enough results, we also came up with a fancier model with 3 variations. The fancier model still uses concatenated two InceptionV3 models, but added an extra dense layer with 128 neurons in between a dense layer with 1024 neurons and a final dense layer with 1 neuron. Variation to this fancier model comes from whether we apply only Batch Normalization, only Dropout, or applying both.

Thus, Models:
Basic model: 2 InceptionV3 models+dense (1024)+desne (1)  (i.e. w/o fine tune, w/o data aug.)
Version a: basic model  with  fine tuning, without data augmentation
Version c: basic model  without fine tuning, with data augmentation
Version d: basic model  with fine tuning, with data augmentation
Fancier model for version 2.x: 2 Inception V3 models +  dense (1024) + dense (128) + desne (1) 
Version 2.a: The fancier model + added only Batch Normalization before each dense layer
Version 2.b: The fancier model + added only Dropout before each dense layer
Version 2.c: The fancier model + added both Batch Normalization and Dropout before each dense layer

Note: for all variations we ran with hyperparameter of 50 for minimum tissue percentage
### Using Zoom level 3,4,5 - stride 150

Initially we started off by using three zoom levels 3,4 and 5. This was an initial model that used a striding window of 150. At this time we didn’t think of balancing the data between tumor and normal patches so there is a much greater number of normal patches compared to tumor patches. Image 78 produced 1548 patches, 36 patches on the x axis and 43 patches on the y axis. 

The predictions were not very good on all the different variants of models we trained. One thing in common was that the models predicted a lot of tumor patches to be normal thus the true positive rate was very low. This could be somewhat expected as our dataset is unbalanced. Also, since, for one example, in zoom level 5, it has dimension of a dimension of 36 x 43,  there are actually not too many images in this zoom level which would have been another reason for the poor performance. 

We thus decided to move on to lower zoom levels such as zoom level 1,2,3,4. As mentioned on the paper that the extra zoom levels (i.e. 3 zoom levels) were not as helpful, we decided to work with only 2 different zoom levels for further progress. This also helped to minimize issues confronting the limited RAM size of what Google Colab Pro provides.

### (Base) Using zoom levels 2,3 - stride 200
We opted to choose level 2 and 3 for this example. Since the zoom level decreases, we opted to have a larger stride of 200 compared to the above example to not produce too many patch images from much larger image sizes. For this example, for test image number 78 and 91, the model version 2.a shows the highest accuracy in that for image number 91, it achieved false negative rate of 0.21, true positive rate of 0.79 and got AUC score of 0.86. However for image number 64, the version c performed better than version 2.a. Moreover, for each image number, the order of which model performs better is not consistent, meaning that while model 1 performs best for one image, there exists other models that perform better for other images. Moreover, highest numeric accuracy results does not provide a holistic measure to decide a best model, since there exists model that achieves good numerical scores, but produced heatmap seem to indicate that it simply over predicted the tumor labels (i.e. for example for biopsy image 91, while model 2.a has higher numeric scores, produced heatmaps from model 2.b. Is more accurate heatmap when compared to original mask label).

###  Using zoom levels 2,3,4 - stride 299

From this example onwards we were no longer able to use data augmentation due to the memory exploding. 

When using zoom levels 2,3,4 Model 2.a. seems to perform the best overall in all three test cases. For image 64, the true positive rate is 0.69 with a AUC score of 0.84. For image 78, the true positive rate is 0.7 with an AUC score of 0.88. And for image 91 the true positive rate is 0.94 with an AUC of 0.88. One thing to note is that although this model seemed to have performed well for image 91, this was not the case. The reason why the true positive rate is so high is because the model over-predicted regions to be tumors. Although we did say that this was better than underpredicting tumorous regions, this might not also be something that the doctor might want. 

### Using zoom levels 2,3,4 - stride 200

This example was the same as the one before except that the stride of the training set was decreased to 200 providing more data to the test set. Unlike the model above, the best model for this example varied quite a bit. The best model for image 62 was Version b, achieving a true positive rate of 0.71 and an AUC score of 0.85. Image 78 was similar to the previous example where version 2.a performed the best with a true positive rate of 0.7 and an AUC score of 0.82. Image 91 didn’t really have a best model as most of the models behaved similarly as the case above where the TPR and AUC were both high due to the model predicting too many patches as tumor. For this example, it is important to note that even with slight change in stride values, the results were quite different and the model shown to be best in the earlier model does not apply the same when different stride value is applied.

### Using zoom levels 1,2 

Unfortunately, for this example we were no longer able to use Image 11 due to memory exploding issues. For image 78 the best model is Version.a. The TPR is 0.71 and the AUC score is 0.85. Image 91 had an TPR of 0.81 and an AUC score of 0.89. The prediction done on Image 91 is the best we can see so far. Although the scores are not too different from the models we used above, the model doesn’t excessively predict patches to be tumorous. 

### Using Zoom levels 0,1

# Conclusion

In conclusion, we believe that the results depend mostly on the resolution of the images and what images we select for the train, test, valid sets. We could observe that the predictions vary on whether the image has a large amount of tumor patches or a small number of tumor patches. The zoom level was also a big factor in the quality of the predictions. The minimum highest zoom level that could predict a decent enough result seemed to be level 4 as it was difficult to extract too many patches from higher zoom levels (i.e. analyzing 3,4,5 did not provide predictions that are good enough). Creating different models by adding features such as data augmentation and more dense layers didn’t help out as much as we wanted it to. The only factor that helped out the model in most cases was the addition of a batch normalization layer.

A potential problem with our evaluation is that the images are not from the same distribution. There were only a few images that had a large percentage of tumors and most of these images were used in the train/test sets. This could be the reason why the validation accuracy only comes out to be 60 percent when the test accuracy often comes out to be much higher. The lack of variety of the tissue images could also be the main reason the models fall short. 

