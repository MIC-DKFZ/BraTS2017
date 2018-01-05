# Contribution to the BraTS 2017 challenge (segmentation only)
We were happy to achieve the third place in 2017's Brain Tumor Segmentation Challenge (BraTS 2017). 
Rankings can be viewed online: http://www.med.upenn.edu/sbia/brats2017/rankings.html

Like it is always the case with code that is written close to deadlines, it was originally not very user friendly. 
We cleaned it up, but it may still not be perfectly clear. If you have any questions please do not hesitate to contact us!
(f.isensee@dkfz.de)

## Prerequisites
Only runs on python 2.7

You need to have the following python packages installed (may be incomplete):

* theano
* lasagne
* numpy
* scipy
* sklearn
* matplotlib
* medpy
* batchgenerators (get it here: https://github.com/MIC-DKFZ/batchgenerators)

You need to have downloaded at least the BraTS 2017 training dataset.

## How to use
### Set paths
We need to know where you saved the data and where you wish to have the results. Go into paths.py and modify the paths 
to suit your setup.


### Data preprocessing
To save computation time during training and prediction we crop all patients to the brain region and normalize via 
z-score. This needs to be done for all datasets (train, val, test):

```
python run_preprocessing -m train
python run_preprocessing -m val
python run_preprocessing -m test
```


### Network training
Now train all five folds of the cross-validation on the training dataset

```
python train_network.py 0
python train_network.py 1
python train_network.py 2
python train_network.py 3
python train_network.py 4

```

After training, do the validation of the cross-validation
```
python validate_network.py 0
python validate_network.py 1
python validate_network.py 2
python validate_network.py 3
python validate_network.py 4

```

So far there is no code that generates a real summary. You will have to do that yourself (sorry)

If you wish to export the training dataset predictions as nifti so that you can upload them to the evaluation platform, use:

```
python save_pred_seg_as_nifti.py -d train -of OUTPUT_FOLDER_FOR_NIFTI
```

### Validation and Test Set
For predicting the validation and test set use the following commands:

```
python predict_val_set.py 0
python predict_val_set.py 1
python predict_val_set.py 2
python predict_val_set.py 3
python predict_val_set.py 4
```
```
python predict_test_set.py 0
python predict_test_set.py 1
python predict_test_set.py 2
python predict_test_set.py 3
python predict_test_set.py 4
```

This will use the five networks from the cross-validation to run the predictions. The predictions can then be 
consolidated and exported as nifti via:

```
python save_pred_seg_as_nifti.py -d val -of OUTPUT_FOLDER_FOR_NIFTI
```

```
python save_pred_seg_as_nifti.py -d test -of OUTPUT_FOLDER_FOR_NIFTI
```