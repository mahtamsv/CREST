# CREST

This is an implementation of CREST in python. CREST is a novel covariance-based method that uses Riemannian and Euclidean geometry and combines spatial and temporal aspects of the feedback-related brain activity in response to error in EEG-based brain-computer interface (BCI) systems. 

This code also provides the implementation of the CSP-CTP algorithm and the DRM-ST algorithm that together make the CREST algorithm. Next is a brief description of each algorithm and how to use it.  

Please note that the code does not perform any extra steps to accomodate for unbalanced train or test input datasets. If you have unbalanced datasets, the code will not encounter any errors but the interpretation of the results may be difficult. Therefore, it is recommended that you simply balance the classes by randomly subsampling the larger class. To minimize loosing data, you can perform balancing multiple times and run the code and then take the average. 


### CREST()
The input data should be of the format [number of frequency bands, number of trials, number of channels, time samples] where the number of freueny bands specifies the number of bandpass filters through which the data is filtered. Note that even if you plan to apply this on a single frequency band, your input data should be of the shape [1, number of trials, number of channels, time samples]. Also, two data structures pertaining to train or test data for classes 1 and 2 should be separately given as input. 

Output measures come in two formats: classification accuracy and AUC. 

The default parameters for CREST are `CREST(numFilt=3, dist_measure='riem',logreg_solver='liblinear')` where `numFilt` indicates the number of CSP and CTP filters for each class, `dist_measure` indicates the distance measure which can be either 'riem' or 'log-eucl', and the solver for the logistic regression which is liblinear by default. 

You can change the solver to any of the available solvers for logistic regression in scikit-learn: ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}. See this link for more information: 

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

```python
from CREST import CREST # CREST.py and its accompanying files should be in your directory 

# call CREST with the default parameters
crest = CREST(numFilt=3, dist_measure='riem')    

# train_data_1 and train_data_2 contain the training data belonging to classes 1 and 2 each with the following format: 
# [number of frequency bands, number of trials, number of channels, time samples]
crest.train(train_data_1, train_data_2)

# test_data_1 and test_data_2 contain the test data belonging to classes 1 and 2 each with the following format: 
# [number of frequency bands, number of trials, number of channels, time samples]
acc_rate = crest.accuracy(test_data_1, test_data_2)
```

### CSP_CTP() and DRM_ST()

These two algorithms are the building blocks of CREST and are also provided as stand-alone classes.
The functions `train`, `accuracy` and `AUC` can similarly be applied to train the classifiers or measure the performance:

```python
from CREST import CSP_CTP

csp_ctp = CSP_CTP(numFilt=3)   
csp_ctp.train(train_data_1, train_data_2)
acc_rate = csp_ctp.accuracy(test_data_1, test_data_2)
```

```python
from CREST import DRM_ST

drm_st = DRM_ST(dist_measure='riem')
drm_st.train(train_data_1, train_data_2)
auc_rate = drm_st.AUC(test_data_1, test_data_2)
```

### Citation

If you use this code, please cite the following paper:

M. Mousavi and V. R. de Sa, "Spatio-temporal analysis of error-related brain activity in active and passive brain–computer interfaces,” BRAIN-COMPUTER INTERFACES, 2019, VOL. 6, NO. 4, 118–127. https://doi.org/10.1080/2326263X.2019.1671040

### Questions/Comments 
Please send any questions or comments to mahta@ucsd.edu or mahta.mousavi@gmail.com

Thank you! 
