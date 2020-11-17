# BraTS2020 Unet3d AutoEncoder

# Data
Available [here](https://www.kaggle.com/awsaf49/brats20-dataset-training-validation). 

All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes.

Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1).

multimodal slices with segmented mask:

<p>
 <img src="https://github.com/mandrakedrink/brats20_Unet3d_AutoEncoder/blob/master/stats/data_sample.svg" width="60%" height="60%">
</p>

3d projections of multimodal scans and segmented mask:
<p>
 <img src="https://github.com/mandrakedrink/brats20_Unet3d_AutoEncoder/blob/master/stats/sample_data.gif" width="60%" height="60%">
</p>

You can also see 3D data projection [here](https://youtu.be/nrmizEvG8aM)

# Formulation of the problem:
+ 1. Each pixel must be labeled “1” if it is part of one of the classes (NCR/NET — label 1, ED — label 2, ET — label 4), and “0” if not.
+ 2. Make a prediction of age and survival days for each unique identifier in the data.

# Solution
+ 1. For automatical segmentation was used Unet3d with group normal layers. - [unet]()
+ 2. To predict age and number of days of survival - the autoencoder was trained to scale the space from 4 * 240 * 240 * 150 to 512, then statistical values, and hidden representations were extracted for each identifier in the data, encoded by the pretrained autoencoder. after wich SVR was trained on this data. - [autoencoder]()

# Result
Unet Result:
<p>
 <img src="https://github.com/mandrakedrink/brats20_Unet3d_AutoEncoder/blob/master/stats/result1.svg" width="40%" height="40%">
</p>
<p>
 <img src="https://github.com/mandrakedrink/brats20_Unet3d_AutoEncoder/blob/master/stats/unet_result.gif" width="60%" height="60%">
</p>

AutoEncoer Result:
<p>
 <img src="https://github.com/mandrakedrink/brats20_Unet3d_AutoEncoder/blob/master/stats/ae_result.gif" width="60%" height="60%">
</p>

More results can be seen [here](https://www.kaggle.com/polomarco/brats20-3dunet-3dautoencoder) or [here](https://www.youtube.com/watch?v=0nliIOj2WVQ).
