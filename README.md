# Music_Recommendation_Through_Facial_Recognition
We aim to develop a CNN model that can classify the images into the following categories: 0 : angry , 1 : disgust , 2 : fear , 3 : happy , 4 : sad 5 : surprise , 6 : neutral and displaying appropriate songs matching with the person emotion.
##  Introduction

  

People tend to express their emotions, mainly by their facial expressions. Music has always been known to alter the mood of an individual. Capturing and recognizing the emotion being voiced by a person and displaying appropriate songs matching the one's mood and can increasingly calm the mind of a user and overall end up giving a pleasing effect.

  

##  Libraries and Frameworks

  

The CNN model was developed using Keras framework of [TensorFlow](https://www.tensorflow.org). We use other Python libraries such as Numpy, Pandas, Matplotlib for image processing, data manipulation, analyzing data, plotting graphs, etc.

  

##  Dataset

  ### Facial Expression Recognition 

The data consists of 48x48 pixel grayscale images of faces. There are a total of 35887 images.


Bar graph representing the number of images of each emotion:


![dataset](https://github.com/abhiraj072/Music_Recommendation_Through_Facial_Recognition/assets/116944692/e25069a2-2609-4728-86c3-900fc762f930)


The dataset was seperated into two parts: a train dataset containing 80% of the images, and a validation dataset containing 20% of the images.

## **Convolutional Neural Networks**

Convolutional Neural Networks (CNNs) are a specialized type of deep learning model that excel in analyzing grid-like data, such as images or sequential data, by leveraging the concept of convolution. CNNs have revolutionized the field of computer vision and achieved remarkable performance in tasks like image classification, object detection, and image segmentation.
  #### **Convolutional Layers**

CNNs employ convolutional layers, which consist of multiple learnable filters or kernels. Each filter performs a convolution operation by sliding across the input data, computing dot products between the filter and local patches of the input. This operation extracts spatial features and learns local patterns and structures. The result is a set of feature maps that capture different aspects of the input data
#### **Pooling Layers**

Pooling layers are used to downsample the feature maps obtained from convolutional layers, reducing the spatial dimensions. The most common pooling operation is max pooling, which selects the maximum value within a defined pooling window. Pooling helps to capture the most important features while reducing the computational complexity and providing some degree of translation invariance.

#### **Fully Connected Layers**

Following the convolutional and pooling layers, CNNs often include fully connected layers, which are traditional neural network layers. These layers connect every neuron in one layer to every neuron in the next layer. Fully connected layers learn high-level representations by combining the features extracted by the convolutional layers. The final fully connected layer produces the output, which can be used for classification, regression, or other tasks.

![238127907-e07d1fd2-0b02-4693-8d04-1921bbfae833](https://github.com/abhiraj072/Music_Recommendation_Through_Facial_Recognition/assets/116944692/08c86c1c-1b37-4b87-ae7a-0f8cfdcdf6e5)

##  Model Architecture

- Added 4 cascading convolutional layers (Conv2D) and pooling layers (MaxPooling2D), defining the number of filters and activation function (ReLU) used.

- Defined 3 fully connected layers, decreasing in number of units, till a 7-member layer is obtained, which is classified using a SoftMax activation.

- Added “DropOut” layers – layers that remove hidden layers of nodes, improving the accuracy of the model.

![model](https://github.com/abhiraj072/Music_Recommendation_Through_Facial_Recognition/assets/116944692/a82d3e5a-7e13-4b83-a4b2-4c7936db2d6e)

 ###  Data Augumentation

Data augumentation is a techinque used to artificially increase the size of the dataset by creating modified copies of the dataset using existing data. The image is modified by adding a random rotation, random shear, random translations, random noise. This makes the model more robust as it has to learn to deal with irregularities in data.
![data_aug](https://github.com/abhiraj072/Music_Recommendation_Through_Facial_Recognition/assets/116944692/a46748ec-75ff-4576-9a1c-74c447366c73)

  ## Result
 - The training accuracy peaked at around 72%, and the validation accuracy stagnated to around 64% by 40 epochs.


![accuracy](https://github.com/abhiraj072/Music_Recommendation_Through_Facial_Recognition/assets/116944692/b4bf0c36-a062-4b30-a11f-ec633f322676)


![loss](https://github.com/abhiraj072/Music_Recommendation_Through_Facial_Recognition/assets/116944692/dfe3642c-f81b-419f-81d2-2e0ac34517cc)

- This is a clear sign of “overfitting” – an undesirable situation wherein the model is unable to improve its training accuracy, normally caused by a data imbalance as in the first dataset plot.

- This is overcome by data augmentation; this will create new images for the model to process and thus improve the accuracy of the model.
   
The output is the emotion of the face expression captured by the input camera feed.
![sad](https://github.com/abhiraj072/Music_Recommendation_Through_Facial_Recognition/assets/116944692/b23f25e0-c9c9-48dc-8424-d2d07a90c226)

![surprise](https://github.com/abhiraj072/Music_Recommendation_Through_Facial_Recognition/assets/116944692/11e62c6e-5a62-4b88-b5dc-7e0e2e5b1030)

  
  ### Song Recommendation
![sadsong](https://github.com/abhiraj072/Music_Recommendation_Through_Facial_Recognition/assets/116944692/9dd97d11-8c12-4ab3-82d7-fef955f17e16)



  

##  Conclusion

  

In conclusion, we have presented a successful approach for the classification of emotion using convolutional neural networks (CNNs). Our proposed model achieved good accuracy and specificity in identifying different facial expressions . Future work should focus on addressing the limitations of our study and exploring other methodologies to further improve the performance of the model.

##  References

  

1.  [Kaggle Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)



3.  [Research Paper](https://www.researchgate.net/publication/351056923_Facial_Expression_Recognition_Using_CNN_with_Keras)
