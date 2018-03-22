# Traffic-Sign-Classifier 

Classify traffic signs using CNNs

## Using Tensorflow and data augmentation - [Approach 1](Traffic_Sign_Classifier.ipynb)

As part of [Udacity's Self Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013), implement LeNet5 model using Tensorflow. This also demonstrates the data augmentation techniques used to avoid overfitting.

## Keras - Finetuning VGG - [Approach 2](Traffic Sign Classifier with Keras.ipynb)
 - Keras with backend as tensorflow.
A lot of time was spent in earlier version while using Tensorflow directly and also defining data augmentation methods.This project demonstrates how one can easily use pretrained models in keras.

A Keras ImageDataGenerator is used for data augmentation techniques and finetuned VGG model by adding dense layer as the last layer.

This project also demonstrates on using keras save and load models.

Even though there was little difference in the accuracies in both methods, it can easily improved by experimenting with other imagenet models too by using Keras.

