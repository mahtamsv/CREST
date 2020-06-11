# CREST

This code provides is an implementation of CREST in python. CREST is a novel covariance-based method that uses Riemannian and Euclidean geometry and combines spatial and temporal aspects of the feedback-related brain activity in response to error in EEG-based brain-computer interface (BCI) systems. 

## How to use 
The input data should be of the format [number of frequency bands, number of epochs, number of channels, time samples]. Note that even if you plan to apply this on a single frequency band, your input data should be of the shape [1, number of epochs, number of channels, time samples]. Also, two data structures pertaining to classes 1 and 2 should be separately given as input. 

Please note that the code assumes balanced inputs classes and it does not perform any extra steps to accomodate for unbalanced train or test input datasets. If you have unbalanced datasets, you can simply balance it multiple times by randomly subsampling the larger class. You can then run the code on balanced classes and take the average. 


### Reference 

If you use this code, please cite the following paper:

M. Mousavi and V. R. de Sa, "Spatio-temporal analysis of error-related brain activity in active and passive brain–computer interfaces,” BRAIN-COMPUTER INTERFACES, 2019, VOL. 6, NO. 4, 118–127. https://doi.org/10.1080/2326263X.2019.1671040
