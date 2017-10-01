# Semantic Segmentation
### Overview
Train a Fully Convolutional Network(FCN) to label the pixels of a road in images.

### Results
Below are some test results of the output. The FCN seems to perform well under normal lighting.
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
