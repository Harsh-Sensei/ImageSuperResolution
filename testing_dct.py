#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import keras
import random
import pandas as pd
import numpy as np
from functools import partial
from math import *
import cv2 as cv
from PIL import Image, ImageOps
from IPython.display import display 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
from scipy import stats
import sklearn 
import sklearn.feature_extraction

from keras import callbacks



def show_bgrimg(image):#for diplaying bgr images
    """
    Displays a BGR image given the array (numpy or otherwise) corresponding
    to the image, using the OpenCV library.
    Args:
        An array of an image.
    Returns:
        A BGR image: The rendition of the image file in the Jupyter Notebook.
    How to be used:
        For example,
        >> arr = cv.imread("butterfly.png")
        >> show_bgrimg(arr)
        will display the image in "butterfly.png" in BGR format.
    """
    plt.figure(figsize=(12, 10), dpi=80)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()


def show_rgbimg(image):#for displaying rgb images
    """
    Displays a RGB image given the array corresponding
    to the image, using the OpenCV library.
    Args:
        An array of an image.
    Returns:
        A RGB image: The rendition of the image file in the Jupyter Notebook.
    How to be used:
        For example,
        >> arr = cv.imread("butterfly.png")
        >> show_rgbimg(arr)
        will display the image in "butterfly.png" in RGB format.
    """
    plt.figure(figsize=(12, 10), dpi=80)
    plt.imshow(image)
    plt.show()



def show_bnw(image):#for displaying grayscale images

    """
    Displays a grayscale image given the array corresponding
    to the image, using the OpenCV library.
    Args:
        An array of an image.
    Returns:
        A grayscale image: The rendition of the image file in the Jupyter Notebook.
    How to be used:
        For example,
        >> arr = cv.imread("butterfly.png")
        >> show_bnw(arr)
        will display the image in "butterfly.png" in grayscale format.
    """
    plt.figure(figsize=(12, 10), dpi=80)
    plt.imshow(image,cmap="gray",vmin=0,vmax=255)
    plt.show()




#function to finc DCT(discrete cosine transform) of an image
def convert_grayscale_to_dct(image):
    """
    Refer documentation for cv.dct and image.astype("float").
    """
    imagefloat=image.astype('float32')
    imagefloat=imagefloat[:,:,0]
    dct_img=np.resize(cv.dct(imagefloat),(image.shape[0],image.shape[1]))
    dct_img=cv.dct(imagefloat)
    return dct_img




#function to convert Img from DCT
def convert_dct_to_grayscale(dct_img):
    """
    Refer documentation for cv.idct.
    """
    dct_img=dct_img[:,:,0].astype('float32')
    image=cv.idct(dct_img)
    return image
    





#function to convert to ycbcr
def conv_ycbcr(image):
    """
    Returns a numpy array corresponding to the YCbCr representation of an image,
    given an array of an image.
    Args:
        An array of an image.
    Returns:
        A luminance channel in the form of numpy array of the image in it's YCbCr representation.
    How to be used:
        >> arr = cv.imread("butterfly.png")
        >> print(conv_ycbcr(arr)) #a,b..z are for representation purposes only.
        [[[a],[b],...,[z]], [[a],[b],...,[z]], ..., [[a],[b],...,[z]]]
        >> print(conv_ycbcr(arr).shape) #Assume "butterfly.png" is a 256x256 image.
        (256,256,1)
    """
    height=image.shape[0]
    width=image.shape[1]
    ycbcr_img=cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
    return np.resize((ycbcr_img[:,:,0]).flatten(),(height,width,1))



def patch_extraction(image,subw,subh):
    """
    Extracts non-overlapping patches of width subw and height subh (discards remaining part)
    Args:
        ndarray, patch width, patch height
    Returns:
        A grayscale image: The rendition of the image file in the Jupyter Notebook.
    How to be used:
        For example,
        >> arr = cv.imread("butterfly.png")
        >> show_bnw(arr)
        will display the image in "butterfly.png" in grayscale format.
    """
    result=[]
    n_x=image.shape[1]//subw
    n_y=image.shape[0]//subh
    for i in range(n_x):
        for j in range(n_y):
            result.append(image[j*subh:(j+1)*subh:1,i*subw:(i+1)*subw:1])
    return result



def bicubic_resize(layer_output,image):
    """
     Returns a numpy array corresponding to an image of dimensions 'layer_output'
     from the image given to it as an input.
     Args:
         A pair of integers, An array of an image.
     Returns:
         A numpy array of the image of the pair of integers.
     How to be used:
         >> arr = cv.imread("butterfly.png")
         >> bicubic_resize((100,100),arr) #a,b..z are for representation purposes only.
         [[[a],[b],...,[z]], [[a],[b],...,[z]], ..., [[a],[b],...,[z]]]
         >> print(bicubic_resize((100,100),arr).shape) #Assume "butterfly.png" is a 256x256 image
         (100,100,1)
     """
    resized_img=cv.resize(image, layer_output, interpolation=cv.INTER_CUBIC)
    resized_img=np.resize(resized_img.flatten(),tuple(list(layer_output)+[1]))
    return resized_img



#function for bilateral filter and bicubic interpolation
def preprocessing(image,scale=2):
    """
    Returns a numpy array corresponding to an image scaled up by 'scale'
    of the image given to it as an input.
    Args:
        An array of an image.
    Returns:
        A numpy array of the scaled up image.
    How to be used:
        >> arr = cv.imread("butterfly.png")
        >> preprocessing(arr) #a,b..z are for representation purposes only.
        [[[a],[b],...,[z]], [[a],[b],...,[z]], ..., [[a],[b],...,[z]]]
        >> print(preprocessing(arr).shape) #Assume "butterfly.png" is a 256x256 image, and
        >> # scale is 2.
        (512,512,1)
    """
    image=cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
    image=image[:,:,0]
    width=image.shape[1]
    height=image.shape[0]
    resized_img=cv.resize(image,(width*scale,height*scale), interpolation=cv.INTER_CUBIC)
    resized_img=np.resize(resized_img.flatten(),(height*scale,width*scale,1))
    resized_img = resized_img.astype('float32')
    return resized_img




def PSNR(original, compressed):
    """
    Returns the Peak Signal to Noise Ratio (PSNR)
    given 2 lists on the basis of the mean squared
    error between them.

    Args:
        2 python lists (y_grdtrth, standing for the "ground Truth"
        and y_pred, standing for the "prediction").
    Returns:
        A float type number
    How to be used:
        >> print(PSNR([[0, 1], [0, 0]], [[1, 1], [0, 0]]))
        0.25
    """
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr




print(sample.shape)
processed_sample=preprocessing(sample)
print(processed_sample.shape)


ds,ds_info = tfds.load('div2k', split='train', shuffle_files=True,with_info=True)
print(type(ds))
print(ds_info)


# How to extract data from a tensorflow dataset: https://www.tensorflow.org/datasets/overview



#extracting images from dataset
train_x=[]
train_y=[]
no_of_images=100
i=0
for sample in ds:
    train_x.append(np.array(sample["lr"]))
    train_y.append(np.array(sample["hr"]))
    i+=1
    if(i==no_of_images):
        break



#shape of the images stored
for i in range(5):
    print(train_x[i].shape,train_y[i].shape)





#initializing parameters
batch_size=32
height=50
width=50



#pre-processing
processed_images_x=list(map(preprocessing,train_x))
grayscale_images_y=list(map(conv_ycbcr,train_y))


#cropping
cropped_images_x=np.array(patch_extraction(processed_images_x[0],height,width))
cropped_images_y=np.array(patch_extraction(grayscale_images_y[0],height,width))
for element in processed_images_x[1:]:
    cropped_images_x=np.append(cropped_images_x,patch_extraction(element,height,width),axis=0)
for element in grayscale_images_y[1:]:
    cropped_images_y=np.append(cropped_images_y,patch_extraction(element,height,width),axis=0)


shape_x=cropped_images_x.shape
shape_y=cropped_images_y.shape


resize=partial(bicubic_resize,(38,38))
resized_cropped_images_y=np.array(list(map(resize,cropped_images_y)))
network_input=np.array(list(map(convert_grayscale_to_dct,cropped_images_x)))
network_output=np.array(list(map(convert_grayscale_to_dct,resized_cropped_images_y)))


#network architecture

network=models.Sequential()
network.add(layers.Conv2D(64, (9,9),activation='relu',input_shape=(height,width,1),name='first'))
network.add(layers.Conv2D(32,(1,1),activation='relu',name='second'))
network.add(layers.Conv2D(1,(5,5),activation='relu',name='last'))



network.summary()


#output dimension of the last layer
last_output=network.get_layer('last').output_shape[1:3]


#network training

network.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                        mode ="min", patience = 5,
                                        restore_best_weights = True)

history = network.fit(network_input[:100000], network_output[:100000],batch_size=batch_size,epochs=30,
                     validation_data =(network_input[100000:105000], network_output[100000:105000]),
                     callbacks =[earlystopping]
                     )


#saving the network
network.save('dct_E30_B32_I100000')


#loading the network
network=models.load_model('dct_E30_B32_I100000')


#checking if the network is loaded correctly
network.summary()



#testing for various input
z=100091
pre=network.predict(network_input[z:z+10])
predict_img=convert_dct_to_grayscale(pre[0]).astype(np.float32)


#displaying the images
show_bnw(predict_img)
show_bnw(cropped_images_x[z])
show_bnw(resized_cropped_images_y[z])


#comparing the PSNR values of predicted/ground truth vs bicubic/ground truth
print(PSNR(resized_cropped_images_y[z],np.resize(predict_img,(predict_img.shape[0],predict_img.shape[1],1))))
print(PSNR(cropped_images_y[z],cropped_images_x[z]))


#MSE evaluation
results = network.evaluate(network_input[105000:],network_output[105000:] , batch_size=128)





