**Traffic Sign Recognition** 
---
**Build a Traffic Sign Recognition Project**
The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Visualization.png "Visualization"
[image2]: ./Grayscale.png "Grayscaling"
[image3]: ./Transform.png "Transform image"
[image4]: ./testdata/image1.jpg "Traffic Sign 1"
[image5]: ./testdata/image2.jpg  "Traffic Sign 2"
[image6]: ./testdata/image3.jpg  "Traffic Sign 3"
[image7]: ./testdata/image4.jpg  "Traffic Sign 4"
[image8]: ./testdata/image5.jpg  "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
You're reading it! and here is a link to my [project code](https://github.com/rock626/project2)

###Data Set Summary & Exploration

I used the python and pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43



Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed per class.

![alt text][image1]

Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color plays a very minor role in classifying traffic signs.For example, in case of speed limit 30, network needs to determine the number whether its color is red or blue or green.By converting to grayscale, we are reducing the number of channels and providing a 2D input instead of 3D.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the range of values in input is minimal , here in this case its -1 to 1. By normalizing, data will have zero mean and equal variance.

I decided to generate additional data because this reduces the problem of overfitting.Initially i trained the model without additional data and found that training accuracy and validation accuracy are not converging.After adding more data, the graph shows that training data is evenly spread across classes.(its not same but better than original training data) 

To add more data to the the data set, I used the following techniques: rotation,translation,shear and brightness.

Here is an example of an original image and an augmented image:

![alt text][image3]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image  						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 35x5	    | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Flatten	      	    | output 1600				                    |
| Fully connected		| output 120        							|
| RELU					|												|
| Dropout	      	    | 0.6 probability 				                |
| Fully connected		| output 84       							    |
| RELU					|												|
| Dropout	      	    | 0.6 probability 				                |
| Fully connected		| output 43       							    |
| Softmax				|         									    |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following hyper parameters
Adam optimizer
batch size :128
EPOCHS : 30
learning rate : 0.001
drop out probaility : 0.6

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 94.8%
* validation set accuracy of 93.9%
* test set accuracy of 94.3%

* I started with LeNet architecture because it's straightforward and small (in terms of memory footprint).Also its taught in the lectures too!!!As its a proven architecture to classify handwritten numbers, it is well suited for the given problem , which is classification of traffic signs.Both are classification problems.
* With just LeNet, my validation accuracy is about 75%. Then i added preprocessing steps including augmentation which improved validation accuracy to 88.5%. A snippet of my notes with different iterations:

LeNet Itr1
Nor : (X-128.)/128.
Augmentation +  Grayscale  + Normalization
15/128
VA/TA - 0.885/0.941
Time = 70 sec

Iteration 2
Augmentation +  Grayscale  + Normalization + dropout 0.5
15/128
Added dropouts at FC , 0.5

VA/TA - 0.83/0.74
104 sec

Iteration 3
Augmentation +  Grayscale  + Normalization +dropout 0.5
50/128
epochs - 50
VA/TA - 0.883/0.845
Time - 351 seconds


Iteration 4
Augmentation +  Grayscale  + Normalization +dropout 0.6
50/128
epochs - 50
VA/TA  - 0.894/0.87
5min 43sec]

Iteration 5:
Augmentation +  Grayscale  + Normalization +dropout 0.6
100/128
100 epochs
VA/TA - 0.915/0.910

Iteration 6:
30 ePOCHS
increase 16 to 32 at conv2, depth
VA/TA 0.901/0.894

Iteration 7:
6 to 12 at conv1,depth
VA/TA - 0.927/0.925

Iteration 8:
drop out - 0.5
VA/TA -0.914/885

Iteration 8:
change depth 32 to 64 , conv2
drop out 0.6
VA/TA : 0.94/0.952

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30       		        | 30  									        | 
| Double curve     		| children crossing 					        |
| Bicycles crossing		| Bicycles crossing								|
| Turn right ahead	    | Turn right ahead					 			|
| Ahead only			| Ahead only     							    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.4%

####3. 
The top five soft max probabilities were

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------:| 
| 9.99786198e-01        | Bicycles crossing  	            | 
| 2.13663297e-04     	| Bumpy Road 						|
| 6.41394351e-08 		| Slippery Road						|
| 5.70957503e-09	    | Children crossing					|
| 2.97234182e-10		| Beware of ice/snow      			|

For the second image :
| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------:| 
| 1.00000000e+00        | Turn Right Ahead  	            | 
| 1.63148428e-08     	| Speed limit 70					|
| 2.79152504e-12 		| Keep left						    |
| 8.38247765e-13	    | Stop					            |
| 2.74561342e-14		| No vehicles     			        |

For the 3rd image :
| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------:| 
| 9.61254418e-01        | Speed limit 30  	                | 
| 1.90812889e-02     	| wild animals crossing 			|
| 8.95221811e-03 		| stop						        |
| 7.05635175e-03	    | Speed limit 20 					|
| 1.44423067e-03		| keep right      			        |

    
For the 4th image :
| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------:| 
| 1.00000000e+00        | Ahead only  	                    | 
| 2.90816714e-13      	| Speed limit 60 					|
| 8.96189128e-16  		| Go straight or right				|
| 7.79668735e-16	    | Turn left ahead					|
| 3.20793773e-19		| Yield     			            |
    
For the 5th image :
| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------:| 
| 9.99613106e-01        | Children crossing  	            | 
| 3.84303334e-04      	| Dangeroous curve to the right 	|
| 2.58285741e-06  		| Slippery Road						|
| 1.51305757e-09	    | Pedestrians					    |
| 6.24303872e-11		| Beware of ice/snow      			|



