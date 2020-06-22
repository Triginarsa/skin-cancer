# Final Project Bangkit : Improved CNN-Based Image Classification


## "Skin Cancer Classification with MobileNetV2"


Dataset: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

<img width="600" alt="lesiondatasets" src="https://user-images.githubusercontent.com/61100613/85271783-72fce000-b4a5-11ea-877c-98a1d324ec81.png">


## Baseline CNN Implementation : MobileNetV1

MobileNets are based on a streamlined architecture that uses depthwise separable convolutions to build lightweight deep neural networks.


## Improvement: Transfer Learning with MobileNetV2, Fine Tuning, Image Augmentation

1.MobileNetV2 

MobileNetV2 builds upon the ideas from MobileNetV1, using depthwise separable convolution as efficient building blocks. However, V2 introduces two new features to the architecture: 1) linear bottlenecks between the layers, and 2) shortcut connections between the bottlenecks1.

2. Fine Tuning

This is a more involved technique, where we do not just replace the final layer (for classification/regression), but we also selectively retrain some of the previous layers. Deep neural networks are highly configurable architectures with various hyperparameters. 

3. Image Augmentation

Image augmentation artificially create training images through different ways of processing or combination of multiple processing, such as random rotation, shifts, shear, and flips, ets.
