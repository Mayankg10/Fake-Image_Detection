# Fake-Image Detection using Deep Learning
The objective of this project is to identify fake images(Fake images are the images that are digitally altered images). The problem with existing fake image detection system is that they can be used detect only specific tampering methods like splicing, coloring etc. We approached the problem using machine learning and neural network to detect almost all kinds of tampering on images.

Using latest image editing softwares, it is possible to make alterations on image which are too difficult for human eye to detect. Even with a complex neural network, it is not possible to determine whether an image is fake or not without identifying a common factor across almost all fake images. So, instead of giving direct raw pixels to the neural network, we gave error level analysed image.

![image](https://user-images.githubusercontent.com/79482851/166126003-a17703fb-0167-4e4f-ac99-bff220facfa1.png)

## Contents

* Methods
  * [1. Error Level Analysis (ELA)](https://github.com/Mayankg10/Fake-Image_Detection#1-Error-Level-Analysis-(ELA))
  * [2. MesoNet](https://github.com/Mayankg10/Fake-Image_Detection#2-MesoNet)
  * [3. Cross Entropy using Neural Network](https://github.com/Mayankg10/Fake-Image_Detection#3-Cross-Entropy-using-Neural-Network)
  * [4. Convolution Neural Network (CNN)](https://github.com/Mayankg10/Fake-Image_Detection#4-Convolution-Neural-Network-(CNN))
  * [5. Generative Adversarial Networks (GAN)](https://github.com/Mayankg10/Fake-Image_Detection#5-Generative-Adversarial-Networks-(GAN))
* [Datasets](https://github.com/Mayankg10/Fake-Image_Detection#Datasets)
* [References](https://github.com/Mayankg10/Fake-Image_Detection#References)

## 1. Error Level Analysis (ELA)

## 2. Mesonet
Due to the increase in popularity of social media and the explosion of technology over
the past decade, images and videos are becoming very common digital objects. With the volume
of digital images available now, different techniques have emerged in ways to alter pictures. This
method tackles one approach to being able to detect a deep-fake image of someone as opposed to
a real image.

Deep-fake images are images in which an existing person’s features can be changed or
replaced by someone else’s likeness

![image](https://user-images.githubusercontent.com/79482851/166126379-74996175-0fcd-4f7c-b194-e3c7492291ee.png)

### Setup Information 
* Dataset
  * The training and testing dataset can be found under the "data" folder
  * For the example of the ipynb my files are saved in a google drive folder at
"/content/drive/My Drive/Colab Notebooks/mesonet/data"
  * To run the same .ipynb file, you must download the data and create a new
folder called ‘mesonet’ with the ‘data’ folder inside of it as well.
* Pretrained Model Weights
  * Pre obtained weights are located in the "weights" folder
  * To speed testing purposes along and for ease of execution, the weights are
located in "/content/drive/My Drive/Colab Notebooks/mesonet/weights"
  * To run the same .ipynb file, you must download the data and create a new
folder called ‘mesonet’ with the ‘weights’ folder inside of it as well.

### Meso-4 Model
The Meso-4 network starts off 4 successive layers of convolution and pooling, which is
followed by one dense layer and then a hidden layer. The convolutional layers use the relu
activation function to help introduce non-linearity into our results. Here is a flowchart example
of the model. The Meso-4 model set up in this way has a total of 27,977 total trainable params.
The model was also supplied with pre-trained weights from the author, these were used to
increase speed of testing and running my program.

![image](https://user-images.githubusercontent.com/79482851/166126448-f33d26fb-6610-4b0a-98cc-ab98afdb595f.png)

### Dataset
The dataset used in the project is a deep-fake dataset created by the makers of the
Meso-net model. The images used to create the deep-fake images were obtained with the mass
amount of digital images and videos flooding the internet nowadays. The authors used the
Viola-Jones detector for extracting features from all the faces, and then a trained neural network
for aligning the features on faces. In the table below is the split of the dataset on the training and
testing images.

Class | Deep Fake | Real
| :--- | ---: | :---:
Deep-fake Training  | 5111 | 7250
Deep-fake Testing  | 2889 | 4259

## 3. Cross Entropy using Neural Network

## 4. Convolution Neural Network (CNN)

## 5. Genetative Adversarial Networks (GAN)

## Datasets

## References
https://github.com/DariusAf/MesoNet
https://www.youtube.com/watch?v=kYeLBZMTLjk
https://towardsdatascience.com/realistic-deepfakes-colab-e13ef7b2bba7
https://hal-upec-upem.archives-ouvertes.fr/hal-01867298/document
