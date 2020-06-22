# Final Project Bangkit : Improved CNN-Based Image Classification
## Baseline CNN Implementation : MobileNetV1

MobileNets are based on a streamlined architecture that uses depthwise separable convolutions to build lightweight deep neural networks.


## Improvement: Transfer Learning with MobileNetV2, Fine Tuning, Image Augmentation

1.MobileNetV2 

MobileNetV2 builds upon the ideas from MobileNetV1, using depthwise separable convolution as efficient building blocks. However, V2 introduces two new features to the architecture: 1) linear bottlenecks between the layers, and 2) shortcut connections between the bottlenecks1.

2. Fine Tuning

This is a more involved technique, where we do not just replace the final layer (for classification/regression), but we also selectively retrain some of the previous layers. Deep neural networks are highly configurable architectures with various hyperparameters. 

3. Image Augmentation

Image augmentation artificially create training images through different ways of processing or combination of multiple processing, such as random rotation, shifts, shear, and flips, ets.



## "Skin Cancer Classification with MobileNetV2"

## IMPORT DATASETS
You can downlaod in https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000 or you can download with api in kaggle

![image](https://user-images.githubusercontent.com/54672242/85274467-2d421680-b4a9-11ea-9e3a-05a340b62854.png)

For detail https://www.kaggle.com/general/51898

## EXPLORE DATASETS
![image](https://user-images.githubusercontent.com/54672242/85273093-56fa3e00-b4a7-11ea-92b3-1544beeec648.png)

Dataset: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000


## PREPROCESSING

1. Separate Train Images
![image](https://user-images.githubusercontent.com/54672242/85274751-91fd7100-b4a9-11ea-8f1d-a3a1c8d0868d.png)

2. Separate Validation Images
![image](https://user-images.githubusercontent.com/54672242/85275801-1270a180-b4ab-11ea-8877-be2822ecb12f.png)

3. Create Augmentation dir then separate to train dir so that the data distribution is evenly distributed
![image](https://user-images.githubusercontent.com/54672242/85276115-81e69100-b4ab-11ea-9490-c65f74ba3dc6.png)

Total Train Images 

![image](https://user-images.githubusercontent.com/54672242/85276576-3e405700-b4ac-11ea-80b6-0e8f7add3d47.png)

Total Val Images

![image](https://user-images.githubusercontent.com/54672242/85276848-a1ca8480-b4ac-11ea-8d8b-c5bb52a7ef51.png)


## BUILD MODEL 

1. Download Mobilenet v2, For Detail you can learn in https://keras.io/api/applications/mobilenet/#mobilenetv2-function

![image](https://user-images.githubusercontent.com/54672242/85277245-46e55d00-b4ad-11ea-8ec3-66b1b3e5af64.png)

2. Add Layers
![image](https://user-images.githubusercontent.com/54672242/85277718-0cc88b00-b4ae-11ea-8708-9f8676040b21.png)

3. Fine Tuning
![image](https://user-images.githubusercontent.com/54672242/85278136-b871db00-b4ae-11ea-9e48-0c0168a85374.png)

for Detail you can learn in https://keras.io/guides/transfer_learning/

## TRAIN MODEL 
![image](https://user-images.githubusercontent.com/54672242/85281306-f45b6f00-b4b3-11ea-88ce-d8fbde4c2de1.png)

## RESULT
![image](https://user-images.githubusercontent.com/54672242/85282103-59639480-b4b5-11ea-87ba-dfe6030a6084.png)

## EVALUATE
1. Classification Report 
![image](https://user-images.githubusercontent.com/54672242/85282302-b8c1a480-b4b5-11ea-9b10-27e815a3d81e.png)

2. Confusion Matrix
![image](https://user-images.githubusercontent.com/54672242/85282362-d0009200-b4b5-11ea-9106-8d640692d623.png)

## CONVERT MODEL TO TENSORFLOWJS
![image](https://user-images.githubusercontent.com/54672242/85282540-1d7cff00-b4b6-11ea-9807-1ee23e659f12.png)

## CREATE WEB
![image](https://user-images.githubusercontent.com/54672242/85282833-99774700-b4b6-11ea-84ca-c67d880a68de.png)

## DEPLOY WEB VIA https://www.netlify.com/
You can learn in https://www.netlify.com/blog/2016/09/29/a-step-by-step-guide-deploying-on-netlify/

## DEMO
1. Download Image from datasets
2. Open Our Web https://dps-4b.netlify.app/
![image](https://user-images.githubusercontent.com/54672242/85283548-c24c0c00-b4b7-11ea-8bf1-5c127c363b14.png)

3. Choose Image File 
![image](https://user-images.githubusercontent.com/54672242/85283699-fe7f6c80-b4b7-11ea-9304-ed43d7a32de3.png)

4. Check The Result Score 
![image](https://user-images.githubusercontent.com/54672242/85283801-28389380-b4b8-11ea-82aa-0ded2f7c1cf1.png)

5. Finish


## THANK TOU :)
