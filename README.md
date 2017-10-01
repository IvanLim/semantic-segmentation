# Semantic Segmentation
### Overview
Train a Fully Convolutional Network(FCN) to label the pixels of a road in images.

### Model overview
There are two parts to a fully convolutional network: the **encoder** that performs the convolutions, and the **decoder** that performs deconvolutions. 

## Encoder
For the encoder, we take advantage of transfer learning, and use the existing **vgg16** pre-trained network. 

From this network, we extract the following:
* The final layer (conv7)
* Pooling layer 3 (pool3)
* Pooling layer 4 (pool4)

## Decoder
Before setting up the decoder, we first apply 1x1 convolutions to all 3 extracted layers. This helps us preserve spatial information.

We then perform the following deconvolution sequence:

**8x_upsample( 2x_upsample(2x_upsample(conv7) + pool4) + pool3 )**

### Results
The FCN seems to perform well under normal lighting.
![Normal example 2](https://github.com/IvanLim/semantic-segmentation/blob/master/report/normal2.png "Normal example 2")

In the case where there is extremely bright light, parts of the road are not properly identified.
![Bright light example](https://github.com/IvanLim/semantic-segmentation/blob/master/report/bright_light.png "Bright light example")

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```
