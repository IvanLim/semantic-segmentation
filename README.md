# Semantic Segmentation
### Overview
Train a Fully Convolutional Network(FCN) to label the pixels of a road in images.

### Model overview
There are two parts to a fully convolutional network: the **encoder** that performs the convolutions, and the **decoder** that performs deconvolutions. 

## Encoder
For the encoder, we take advantage of transfer learning, and use the existing **VGG16** pre-trained network. 

From this network, we extract the following:
* The final layer (conv7)
* Pooling layer 3 (pool3)
* Pooling layer 4 (pool4)

## Decoder
Before setting up the decoder, we first apply 1x1 convolutions to all 3 extracted layers. This helps us preserve spatial information.

We then perform the following deconvolution sequence:

**8x_upsample( 2x_upsample(2x_upsample(conv7) + pool4) + pool3 )**

## Weights initialization, and regularization
When initializing the weights, I originally tried initializing with a **Xavier initializer**, but the results were terrible. After several rounds of experimentation with different combinations, I achieve the best results **without** using L2 regularization and by initializing the weights with a standard **truncated normal initializer**.

### Results
The FCN seems to perform well under different cases: single roads, double roads, and in cases where there are strong shadows.

![One road in image](https://github.com/IvanLim/semantic-segmentation/blob/master/report/single.png "One road in image")

![Two roads in image](https://github.com/IvanLim/semantic-segmentation/blob/master/report/double.png "Two roads in image")

![Strong shadows](https://github.com/IvanLim/semantic-segmentation/blob/master/report/shadows.png "Strong shadows")


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
