# Fake-Image Detection using Deep Learning
The objective of this project is to identify fake images(Fake images are the images that are digitally altered images). The problem with existing fake image detection system is that they can be used detect only specific tampering methods like splicing, coloring etc. 

We approached the problem using machine learning and neural network to detect almost all kinds of tampering on images.

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
 
Many images are spread in the virtual world of social media. With the many editing software that allows so there is no doubt that many forgery images. By forensic the image using Error Level Analysis to find out the compression ratio between the original image and the fake image, because the original image compression and fake images are different. In addition to knowing whether the image is genuine or fake can analyze the metadata of the image, but the possibility of metadata can be changed. 

In this case we apply Deep Learning to recognize images of manipulations through the dataset of a fake image and original images via Error Level Analysis on each image and supporting parameters for error rate analysis. The result of our experiment is that we get the best accuracy of training 98.8% and 94.4% validation by going through 30 epoch.

### Setup

* Dataset
  * The training and testing dataset can be found on the Kaggle site: https://www.kaggle.com/datasets/sophatvathana/casia-dataset
  * The file path for my downloaded data is under "/content/drive/MyDrive/Colab Notebooks/Image_Detector/CASIA2/" which contains two folders, one for real images and one for fake images.
  * To run the same .ipynb file, you must download the data, change all the data file path and simply run the file.
* Model Training 
  * All the model training are saved as a seperate file called "model_casia_run1.h5"
  * To run the same .ipynb file, you must download the data and create a new
folder called ???CASIA1??? with the "model_casia_run1.h5" file called in the function.

### Model

![image](https://user-images.githubusercontent.com/79482851/166129029-b34433f1-06ef-4605-9e31-006c8806fba7.png)

![image](https://user-images.githubusercontent.com/79482851/166129033-d7aaa6d7-b0b6-4c0d-8926-3ccea4fcc897.png)

### Dataset

The Error Level Analysis method uses a kaggle dataset called CASIA1 which has over 7491 real images and 5123 fake images of real world objects. This dataset was perfect for this method as we can test the Machine/Deep Learning model on the real world images to classify the images as fake or real.

Class | Fake | Real
| :--- | ---: | :---:
Training  | 5123 | 7491
Testing  | 21000 | 4100

### Results

By using ELA and CNN model, I was able to get 13854 real images correct out of 14708 images with a 94.4% accuracy rate and 2092 fake images out of 2117 images with a accuracy rate of 98.8%.  

Model | Fake | Real | Total
| :--- | ---: | :---: | :---:
CNN  | 98.8% | 94.1% | 96.5%

## 2. Mesonet
Due to the increase in popularity of social media and the explosion of technology over
the past decade, images and videos are becoming very common digital objects. With the volume
of digital images available now, different techniques have emerged in ways to alter pictures. This
method tackles one approach to being able to detect a deep-fake image of someone as opposed to
a real image.

Deep-fake images are images in which an existing person???s features can be changed or
replaced by someone else???s likeness

![image](https://user-images.githubusercontent.com/79482851/166126379-74996175-0fcd-4f7c-b194-e3c7492291ee.png)

### Setup
* Dataset
  * The training and testing dataset can be found at https://drive.google.com/drive/folders/1YIGj57VJJ3rbgTubkhO9YaTQFsqRHet_?usp=sharing
  * Images are saved in the 'data' folder
  * For the example of the ipynb my files are saved in a google drive folder at
"/content/drive/My Drive/Colab Notebooks/mesonet/data"
  * To run the same .ipynb file, you must download the data and create a new
folder called ???mesonet??? with the ???data??? folder inside of it as well.
* Pretrained Model Weights
  * Pre obtained weights are located at https://drive.google.com/drive/folders/1YIGj57VJJ3rbgTubkhO9YaTQFsqRHet_?usp=sharing
  * You will need to download the folder called 'weights'
  * To speed testing purposes along and for ease of execution, the weights are
located in "/content/drive/My Drive/Colab Notebooks/mesonet/weights"
  * To run the same .ipynb file, you must download the weights and create a new
folder called ???mesonet??? with the ???weights??? folder inside of it as well.

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

### Results
Out of the 19,509 total images the model was able to get the accuracies depicted below.
Although in the .ipynb file in the repository, the accuracy reported is different in order to be in
alignment with the dataset size of other group members.

Class | Deep Fake | Real | Total
| :--- | ---: | :---: | :---:
Meso-4  | 88.2% | 90.1% | 89.1%


## 3. Cross Entropy using Neural Network
A feedforward neural network performs very well especially in image processing. A neural networks evolved from the multilayer perceptron (MLP). neural networks have three structural characteristics: local connection, weight sharing and downsampling. Weight
sharing makes the network structure of neural networks more similar to biological neural networks.
Local connections means that each neuron in layer n-1 is connected to all neurons in layer n, but
between neurons in layer n-1 and some neurons in layer n. This feature helps speed up the file
transfer between neurons in each layer.

### Setup
* Dataset
  * For collecting data, I chose the Mnist database that can be downloaded
directly on the Internet. This is a database that provides handwritten data
sets, so that we can directly perform deep learning of neural networks.
  * I choose python to train my neural network, because python can easily
call the mnist database and numpy database.
  * To install the MNIST dataset in my method ,I just need to type the order below:
!pip install mnist
* The structure of Neural network
  * For Training a neural network generally consists of two stages:
forward phase: Input parameters are passed through the entire network.
backward phase: Backpropagation updates gradient and weight.
  * In the forward phase, each layer needs to store some data (such as input data,
intermediate values, etc.). These data will be used in the backward phase.
  * In the backward phase, each layer gets the output gradient and also
returns the gradient as input. What is obtained is the gradient of loss for
the output of this layer, and what is returned is the gradient of loss for the
input of this layer.I improved the whole system by using cross entropy
function

### Model
The first layer:
I built a sliding window in the first layer that slides back and forth with a specific step size
across the input image. After operation, we get the feature map of the input image, which is the local
feature extracted by the first layer, and the kernel shares parameters. During the training process of the
entire network, the kernels containing the weights are also continuously updated until the training is
completed. The first layer has multiple cores, because weight sharing means that each core can only
extract one feature, so in order to improve the computing power of the entire system, multiple cores
need to be set.

![image](https://user-images.githubusercontent.com/79482851/166128329-c87ea59b-870d-4a4a-b7d3-93340ef0f30a.png)

The main function of the second layer is to reduce the dimensionality of the output of the first
layer. Dimensionality reduction is to remove redundant information, compress features, simplify
network complexity, reduce computation, reduce memory consumption.

The third layer I call the softmax layer, Softmax turns arbitrary real values into probabilities.
This is also the most critical step in image judgment, because this layer converts the values calculated
by the upper two layers into probabilities to determine whether the image is correct.

The cross-entropy loss function is introduced in the backward phase. The definition of the
cross-entropy loss function is the gap between the two probabilities. Through this function, the
probability I get through the forward pass and the actual predicted probability can be well obtained.
The gap between them can be updated with more accurate gradient and weight through backward
calculation.

Then summing forward pass and backward pass I can get a complete neural network.

### Dataset
The MNIST database of handwritten digits, available from this page, has a training set of
60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition
methods on real-world data while spending minimal efforts on preprocessing and formatting.

![image](https://user-images.githubusercontent.com/79482851/166128352-fc6602ee-c242-4a36-874b-96a7175d1b7b.png)

### Results

![image](https://user-images.githubusercontent.com/79482851/166128360-9b4ae2e2-a3de-4702-9da9-70c23e067545.png)


## 4. Convolution Neural Network (CNN)

A convolutional neural network is a type of neural network that is most often applied to image processing
problems. We???ve probably seen them anywhere a computer is identifying objects in an image, but you
can also use CNNs in natural language processing projects too. The fact that they are useful for these fastgrowing
areas is one of the main reasons they???re so important in deep learning and artificial intelligence
today.

Let???s first take a regular neural network. A regular neural network has an input layer, hidden layers, and
an output layer. The input layer accepts input in different forms while the hidden layers perform
calculations on these inputs. The output layer then delivers the outcome of the calculation and
extractions. Each of these layers contains neurons that are connected to neurons in the previous layer
and each neuron has its own weight. This means you aren???t making assumptions about the data being fed
into the network. Great usually but not if you???re working with images or language.

![image](https://user-images.githubusercontent.com/79482851/166132703-1aa83169-7d25-49c9-b78b-bf37f7335464.png)

CNNs work differently as they treat data as spatial. Instead of neurons being connected to every neuron
in the previous layer, they are instead only connected to neurons close to it and all have the same weight.
The simplification in the connections means the network upholds the spatial aspect of the data set.

The word convolutional refers to the filtering process that happens in this type of network. Think of it this
way, an image is complex. A convolutional neural network simplifies it, so it can be better processed and
understood.

### Setup

* Dataset
  * The training and testing dataset can be found under the "data" folder
  * For the example of the ipynb my files are saved in a google drive folder at "/content/drive/My Drive/Colab Notebooks/Artificial Intelligence/Data/dataset"
  * To run the same .ipynb file, you must download the data and create a similar directory structure.

The dataset has been obtained from Kaggle with the dataset called Real and Fake Face Detection. It has
been published by the Computational Intelligence and Photography Lab.
Department of Computer Science, Yonsei University
https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection

### Model

The ReLU layer acts as an activation function, ensuring non-linearity as the data moves through each layer
in the network. Without it, the data being fed into each layer would lose the dimensionality that we want
to maintain. The fully connected layer meanwhile allows you to perform classification on your dataset.

![image](https://user-images.githubusercontent.com/79482851/166132770-b2a210b7-e7e1-4799-b84d-8f0d9a7e365a.png)

CNN is the most important, so let???s start there.
It allows by placing a filter over an array of image pixels. This then creates what???s called a convolved
feature map. It???s a bit like looking at an image through a window, which allows you to see specific features
you might not otherwise be able to see.

![image](https://user-images.githubusercontent.com/79482851/166132795-70835129-6b73-4c8c-b4e8-dd64a3d23be1.png)

Next, we have the pooling layer. This down samples or reduces the sample size of a particular feature
map. This also makes processing much faster as it reduces the number of parameters the network needs
to process. The output of this is a pooled feature map. There are two ways to do this: max pooling, which
takes the maximum input of a convolved feature, or average pooling, which simply takes the average.

These steps amount to feature extraction whereby the network builds up a picture of the image data
according to its own mathematical rules. If you want to perform classification, you???ll need to move into
the fully connected layer. To do this, you???ll need to flatten things out. Remember, a neural network with
a more complex set of connections can only process linear data.

![image](https://user-images.githubusercontent.com/79482851/166132748-d1d44aed-1ea7-45b7-9d05-ee4a98777f53.png)

There are number of ways you can train a CNN. If you are working with unlabeled data, you can use
unsupervised learning methods. One of the best popular ways of doing this is autoencoders. This allows
you to squeeze data in a space with low dimensions, performing calculations in the first part of the CNN.
Once this is done, you???ll then need to reconstruct with additional layers that up sample the data you have.

### Results

![image](https://user-images.githubusercontent.com/79482851/166132740-08af4bab-373c-432c-b933-18f21605027f.png)

## 5. Genetative Adversarial Networks (GAN)

Generative Adversarial Networks, or GANs for short, are an approach to generative modeling
using deep learning methods, such as convolutional neural networks.

Generative modeling is an unsupervised learning task in machine learning that involves
automatically discovering and learning the regularities or patterns in input data in such a way
that the model can be used to generate or output new examples that plausibly could have been
drawn from the original dataset.

GANs are a clever way of training a generative model by framing the problem as a supervised
learning problem with two sub-models: the generator model that we train to generate new
examples, and the discriminator model that tries to classify examples as either real (from the
domain) or fake (generated). The two models are trained together in a zero-sum game,
adversarial, until the discriminator model is fooled about half the time, meaning the generator
model is generating plausible examples.

GANs are an exciting and rapidly changing field, delivering on the promise of generative models
in their ability to generate realistic examples across a range of problem domains, most notably
in image-to-image translation tasks such as translating photos of summer to winter or day to
night, and in generating photorealistic photos of objects, scenes, and people that even humans
cannot tell are fake.

#### Unsupervised Learning

![image](https://user-images.githubusercontent.com/79482851/166132026-690d9706-9166-48fb-b379-9c11cb5a69a0.png)

#### Supervised Learning

![image](https://user-images.githubusercontent.com/79482851/166132043-08bbdd05-7383-40da-a162-ce022f6f467b.png)

#### Generative Model

![image](https://user-images.githubusercontent.com/79482851/166132055-d4fe22b9-60b4-4d76-aeff-d3f3968e25f5.png)

### Setup

* Generator:
  * Given input of noise (latent) vector, the Generator produces an image.
  * I???m only using Dense layers. But the network can be complicated based
on the application.
  * Given an input image, the Discriminator outputs the likelihood of the image
being real.
  * Constructed two models and put them against each other.
  * I do this by defining a training function, loading the data set, re-scaling our
training images and setting the ground truths.
* Data Set:
  * The training and testing dataset can be found under the "data" folder ??? For
the example of the ipynb my files are saved in a google drive folder at
"/content/drive/My Drive/Colab Notebooks/ArtificialIntelligence/images" To
run the same .ipynb file, you must download the data and create a new
folder called ???fakeImage with the ???images??? folder inside of it as well.

### Model

* GANs typically work with image data and use Convolutional Neural
Networks, or CNNs, as the generator and discriminator models.
* The reason for this may be both because the first description of the
technique was in the field of computer vision and used CNNs and image
data, and because of the remarkable progress that has been seen in
recent years using CNNs more generally to achieve state-of-the-art results
on a suite of computer vision tasks such as object detection and face
recognition.
* Modeling image data means that the latent space, the input to the
generator, provides a compressed representation of the set of images or
photographs used to train the model. It also means that the generator
generates new images or photographs, providing an output that can be
easily viewed and assessed by developers or users of the model.
* It may be this fact above others, the ability to visually assess the quality of
the generated output, that has both led to the focus of computer vision
applications with CNNs and on the massive leaps in the capability of
GANs as compared to other generative models, deep learning based or
otherwise.

![image](https://user-images.githubusercontent.com/79482851/166132098-2ea3d6df-0fb0-4af9-be56-40a9b076e378.png)

## References
https://github.com/DariusAf/MesoNet

https://www.youtube.com/watch?v=kYeLBZMTLjk

https://towardsdatascience.com/realistic-deepfakes-colab-e13ef7b2bba7

https://hal-upec-upem.archives-ouvertes.fr/hal-01867298/document

[1] Yadav, Suman ; Saxena, Manish., ???A Novel VLSI Architecture for Reversible ALU Logic
Gate Structure??? International journal of computer applications, 2015-06-18,Vol.120 (2), p.1-3

[2] Cheng, Long ; Li, Yi ; Yin, Kang-Sheng , ???A self-organizing neural network model for a
mechanism of visual pattern recognition[M]??? Competition and cooperation in neural nets, Springer,
Berlin, Heidelberg, pp.267-285. 1982

[3] LeCun Y, Boser B, Denker J S, et al, ???Back propagation applied to handwritten zip code
recognition[J],??? Neural com-putation, vol. 1, no. 4, pp. 541-551, Sep. 19

https://www.kaggle.com/datasets/sophatvathana/casia-dataset

https://github.com/agusgun/FakeImageDetector

https://youtu.be/uqomO_BZ44g

https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/

https://github.com/agusgun/FakeImageDetector

https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection 

https://www.youtube.com/watch?v=jwpSMg6Ebp0

https://github.com/agusgun/FakeImageDetector

[Back to Top](#Fake-Image-Detection-using-Deep-Learning)
  <a name="Fake-Image-Detection-using-Deep-Learning"></a>    
