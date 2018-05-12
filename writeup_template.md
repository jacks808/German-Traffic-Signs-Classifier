

# Traffic Sign Recognition 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jacks808/German-Traffic-Signs-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?

```
34799
```

* The size of the validation set is ?

```
4410
```

* The size of test set is ?

```
12630
```

* The shape of a traffic sign image is ?

```
32, 32, 3
```

* The number of unique classes/labels in the data set is ?

```
43
```



#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set with pandas. It is a bar chart showing how the data distribution. In some label the data is less than 200. Maybe need some image gereration tech to make sure data in every label as almost have same size.

![](https://ws3.sinaimg.cn/large/006tKfTcly1fqx1h32dobj30ot099q38.jpg)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At the begining, I explore the image data distribution use pandas.

![](https://ws3.sinaimg.cn/large/006tKfTcly1fqx1h32dobj30ot099q38.jpg)

As a first step, I decided to convert the images to grayscale because color information is not relay useful for this task, but color information will take more than 3 times data. So I can decrease the weight number for the input but not change the image label. 

![](https://ws2.sinaimg.cn/large/006tKfTcly1fqx1ltbs4hj309i07mq39.jpg)

And because the image data range is from `0` to `255`. which is not good for neural network traing. so I use normalization to move the image data to near `0`. 



Here is a simple to show why normailzation is better way:

![](https://ws2.sinaimg.cn/large/006tNc79ly1fr8eojho9rj316u0vwn31.jpg)



```python
# normalization
X_train_hanlded = (X_train_hanlded - 128)/128
X_test_hanlded = (X_test_hanlded - 128)/128
```

Then the mean of data is reduce from `82` to `-0.35`

![](https://ws4.sinaimg.cn/large/006tKfTcly1fqx1ohwqalj30cj029jrl.jpg)




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Gray image   					|
| Convolution 5x5    | 5x5 stride,6 convolution filter, VALID padding, outputs 28x28x6 |
| Max pooling	| 2x2 stride,  outputs 14x14x6 |
| Convolution 5x5	| 5x5 stride,16 convolution filter, VALID padding, outputs 10x10x16 |
| Max pooling	| 2x2 stride,  outputs 5x5x16 |
| Fully connected		| input 5x5x16 = 400, outputs 120, drop out 50% |
| Fully connected	| input 120, outputs 84, drop out 50% |
| Fully connected | Input 84, outputs 43 |
|						|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an `AdamOptimizer` to optimize the neural network. I try many hyperparameters to turn. 

Batch size: I choose different batch size, such as 100, 128, 256, 512 etc. more large batch size I use, less training loop I need. but I found the gpu memory is not enough to set a very large batch size. For example, A image with `32x32x1` need `32x32x1 = 1024` float value at input layer(for save a image), need `5x5x6=150` float weight and `6` float bias at conv1 layer etc. So I finally use `128` as my batch size.

Epochs: Epoch means the total loop for all data in training. Less epoch makes model underfitting, more epoch makes overfitting. To prevent underfitting, I choose `100` as my epoch, to prevent overfitting, I choose `50%` drop out rate. 

Learning rate: big learning rate make model learning faster than small learning, but the model maybe trapped around some minimum gradient value. After some experiment, I choose `0.001` as my learning rate.

 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of ? 

`95.33%`

* test set accuracy of ?

`94.355%`

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I choose origin `LeNet` to make my hand dirty, because `LeNet` is a traditional convolution neural network for image classification. 

* What were some problems with the initial architecture?

With the original `LeNet` I got this pipeline working. but after several epoch, the overfitting problem happens. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

After some test on this model. change the 2nd conv layer from 3x3 to 5x5. So the second layer can contains more information. And add drop out to the last 2 fully connected layer.

* Which parameters were tuned? How were they adjusted and why?

Add drop out feature for fix over fitting problem. And choose drop out rate in 10%, 30%, 50%. After some test. I finally use 50% drop out rate. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The convolution neural network was well design for image classification. Because convolution filter are share parameter, so the conv net don't need to face huge parameter problem. And conv net use pooling technique to choose the most importent feature from each conv layer output and reduce parameter number. 

Thanks for the drop out technique, the network doesn't rely on some neural. Any neural could be drop when the network is training. 

If a well known architecture was chosen:
* What architecture was chosen?


A convolution neural network, like `LeNet`

* Why did you believe it would be relevant to the traffic sign application?


Because the `LeNet` take `28x28x1` image as it's input, take `10` classes as it's output. 

Both `MNIST` and `Traffice sign` are image classification problem.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


The accuracy of training is about 95%, This mean this model learned the feature of data. Test accuracy about 94% means this model didn't overfitting on training data.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web:

![-](collected-traffic-sign-data/0-1.png)

![-](collected-traffic-sign-data/3-2.png)

![-](collected-traffic-sign-data/4-1.png)

![-](collected-traffic-sign-data/8-1.png)

![1-](collected-traffic-sign-data/11-1.png)

![4-](collected-traffic-sign-data/14-1.png)

![8-](collected-traffic-sign-data/28-1.png)

![4-](collected-traffic-sign-data/34-1.png)





The 3rd image might be difficult to classify because this image is very dark( the mean value of this image is `18.71`). This means after normalization `(image - 128) / 128` small than other images.



![Origin image mean bar chart](https://ws3.sinaimg.cn/large/006tKfTcly1fqxcm6t8c0j30vy0n875b.jpg)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (20km/h) | Speed limit (20km/h) |
| Speed limit (50km/h) | Speed limit (50km/h) |
| Speed limit (70km/h)	| Speed limit (70km/h)	|
| Speed limit (120km/h)	| Speed limit (120km/h)	|
| Right-of-way at the next intersection	| Right-of-way at the next intersection |
| Stop	| Stop |
| Children crossing	| Children crossing |
| Turn left ahead	| Turn left ahead |


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![](https://ws2.sinaimg.cn/large/006tKfTcly1fqxd7ufl36j30po0tg42b.jpg)

![](https://ws2.sinaimg.cn/large/006tKfTcly1fqxd8l16myj30q40tqadv.jpg)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



![](https://ws4.sinaimg.cn/large/006tKfTcly1fqxd90sg38j312808k40c.jpg)


