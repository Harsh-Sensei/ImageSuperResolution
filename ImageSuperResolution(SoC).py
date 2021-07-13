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
    plt.imshow(image,cmap="gray")
    plt.show()







def PSNR_tf(y_grdtrth,y_pred):
    """
     Returns the Peak Signal to Noise Ratio (PSNR)
     given 2 lists on the basis of the mean squared
     error between them.(using tensorflow)

     Args:
         2 python lists (y_grdtrth, standing for the "ground Truth"
         and y_pred, standing for the "prediction").
     Returns:
         A float type number

     """
    m = tf.keras.metrics.MeanSquaredError()
    m.update_state(y_grdtrth,y_pred)
    MSE=m.result().numpy()
    
    result=10*log10((255**2)/MSE)
    return result



def PSNR(original, compressed):
    """
     Returns the Peak Signal to Noise Ratio (PSNR)
     given 2 lists on the basis of the mean squared
     error between them.(using numpy)

     Args:
         2 python lists (y_grdtrth, standing for the "ground Truth"
         and y_pred, standing for the "prediction").
     Returns:
         A float type number

     """
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr




#sample image
sample=cv.imread('sage.png')


#function to finc DCT(discrete cosine transform) of an image
def convert_img_to_dct(image):
    """
    Refer documentation for cv.dct and image.astype("float").
    """
    imagefloat=image.astype("float")
    dct_img=cv.dct(imagefloat)
    return dct_img

#function to convert Img from DCT
def convert_dct_to_img(dct_img):
    """
    Refer documentation for cv.idct.
    """
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








#loading datasets
tfds.list_builders()



ds,ds_info = tfds.load('div2k', split='train', shuffle_files=True,with_info=True)
print(type(ds))
print(ds_info)



#to show the functions of "ds"
print(dir(ds))




#testing if the dataset has loaded
for example in ds.take(1):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
  print(list(example.keys()))
  hr_img = example["hr"]
  lr_img = example["lr"]
  print(hr_img.shape, lr_img.shape)
show_rgbimg(np.array(hr_img))
show_rgbimg(np.array(lr_img))


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
batch_size=30
height=50
width=50



#pre-processing
processed_images_x=list(map(preprocessing,train_x))
grayscale_images_y=list(map(conv_ycbcr,train_y))




show_bnw(processed_images_x[1])
show_bnw(grayscale_images_y[1])



#cropping
cropped_images_x=np.array(patch_extraction(processed_images_x[0],height,width))
cropped_images_y=np.array(patch_extraction(grayscale_images_y[0],height,width))
for element in processed_images_x[1:]:
    cropped_images_x=np.append(cropped_images_x,patch_extraction(element,height,width),axis=0)
for element in grayscale_images_y[1:]:
    cropped_images_y=np.append(cropped_images_y,patch_extraction(element,height,width),axis=0)

    
shape_x=cropped_images_x.shape
shape_y=cropped_images_y.shape


normalized_cropped_images_x=np.resize(stats.zscore(cropped_images_x.flatten()),shape_x)


normalized_cropped_images_y=cropped_images_y/255









#network architecture

network=models.Sequential()
network.add(layers.Conv2D(64, (9,9),activation='relu',input_shape=(height,width,1),name='first'))
network.add(layers.Conv2D(32,(1,1),activation='relu',name='second'))
network.add(layers.Conv2D(1,(5,5),activation='relu',name='last'))



#summary of achitecture
network.summary()


#output dimensions of the last layer
last_output=network.get_layer('last').output_shape[1:3]



resize=partial(bicubic_resize,last_output)
resized_normalized_cropped_images_y=np.array(list(map(resize,normalized_cropped_images_y)))



#network training
network.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])
history = network.fit(normalized_cropped_images_x[:100000], resized_normalized_cropped_images_y[:100000],batch_size=30,epochs=20)

#saving network
network.save('without_dct_E20_B30_I100000')


#loading the saved network
network=models.load_model('without_dct_E20_B30_I100000')

#checkin if network is loaded
network.summary()

#getting the predictions
predictions=network.predict(normalized_cropped_images_x[100000:])


#testing
z=105010
print("Processed input")
show_bnw(np.resize(cropped_images_x[z].flatten(),(height,width)))
print("Predicted")
show_bnw(np.resize(predictions[z-100000].flatten(),last_output))
print("Ground Truth")
show_bnw(np.resize((resized_normalized_cropped_images_y[z]*255).flatten(),last_output))



#evaluation
results = network.evaluate(normalized_cropped_images_x[100000:],resized_normalized_cropped_images_y[100000:], batch_size=30)
print(results)



#final class for super resolution

class SuperResolution:
    """
    Class for predicting the output image of an input array (image) based on a
    trained CNN ('without_dct_E20_B30_I100000') above.
    Args:
        image: An array of an image.
        type : How the output is to be rendered. Default setting is 'bgr'.
        scale : The upscale factor of resolution. Default setting is 2,
        but no point in changing it as this is a fixed hyperparameter of our CNN.
    """
    def __init__(self, image, type='bgr',scale=2):
        self.image=image
        self.network=models.load_model('without_dct_E20_B30_I100000')
        self.shape=image.shape
        self.input_dim=(50,50)
        self.type=type
        self.scale=2
        self.n_x=(self.image.shape[1]*self.scale)//(50)
        self.n_y=(self.image.shape[0]*self.scale)//(50)
        self.output=None
        self.output_cb=None
        self.output_cr=None
        self.output_y=None
        self.predictions=None

        
    #displaying functions
    def show_bgrimg(self,image):#for diplaying bgr images
        plt.figure(figsize=(12, 10), dpi=80)
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.show()
    def show_rgbimg(self,image):#for displaying rgb images
        plt.figure(figsize=(12, 10), dpi=80)
        plt.imshow(image)
        plt.show()
    def show_bnw(self,image):#for displaying grayscale images 
        plt.figure(figsize=(12, 10), dpi=80)
        plt.imshow(image,cmap="gray")
        plt.show()
    
    #summary of model to be used for super resolution
    def model_summary(self):
        self.network.summary()
    
    #dividing the given image into sub-images
    def __patch_extraction(self,image,stride=0):
        """
        Returns a numpy array of patches of the image.
        Args:
            self, image
        Returns:
            A numpy array of patches of the image.
        How to be used:
            For internal usage by the class's methods. Not part of the API interface.
        """
        result=[]
        print("image",image.shape)
        resized_img=cv.resize(image, ((image.shape[1]//50)*50,(image.shape[0]//50)*50), interpolation=cv.INTER_CUBIC)
        resized_img=np.resize(resized_img,((image.shape[0]//50)*50,(image.shape[1]//50)*50,1))
        print("cv",resized_img.shape)
        print(self.n_x)
        if stride==0:
            for i in range(image.shape[0]//50):
                for j in range(image.shape[1]//50):
                    result.append(resized_img[i*50:(i+1)*50,j*50:(j+1)*50])
        else:
            for i in range((image.shape[0]//50)*2-1):
                for j in range((image.shape[1]//50)*2-1):
                    result.append(resized_img[i*25:i*25+50,j*25:j*25+50])
        return np.array(result)
    
    
    def __preprocessing(self):
        ycrcb_img=None
        if self.type=='bgr':
            ycrcb_img=cv.cvtColor(self.image, cv.COLOR_BGR2YCR_CB)
        elif self.type=='rgb':
            yrbcb_img=cv.cvtColor(self.image, cv.COLOR_RGB2YCR_CB)
        else:
            assert(False)

        width=self.shape[1]
        height=self.shape[0]
        
        luminance=ycrcb_img[:,:,0]

        self.output_cr=cv.resize(ycrcb_img[:,:,2],(width*self.scale,height*self.scale),interpolation=cv.INTER_CUBIC)
        self.output_cb=cv.resize(ycrcb_img[:,:,1],(width*self.scale,height*self.scale),interpolation=cv.INTER_CUBIC)
        
        resized_img=cv.resize(luminance,(width*self.scale,height*self.scale), interpolation=cv.INTER_CUBIC)
        resized_img=np.resize(resized_img.flatten(),(height*self.scale,width*self.scale,1))
        resized_img = resized_img.astype('float32')
        print("resized",resized_img.shape)
        
        return resized_img
        
    def __reconstruction(self,patches):
        img_width=self.n_x*38
        img_height=self.n_y*38
        result=np.zeros((img_height,img_width))
        try:
            for i in range(self.n_y*2-1):
                for j in range(self.n_x*2-1):
                    req_patch=patches[i*(self.n_x*2-1)+j]
                    for y in range(38):
                        for x in range(38):
                            result[i*19+y,j*19+x]+=req_patch[y,x,0]
                            
        except IndexError: 
             print("Index Error")
        return result/4
        
    def __thresholding(self,image,max_intensity):
    """
        Returns an ndarray with all the elements greater thn max_intesity clipped to max_intensity

        Args:
            image: ndarray  max_intensity: required max value where elements are to be clipped
        Returns:
            An array of the predicted super-resolved image

    """
        i,j=0,0
        for rows in image:
            i=0
            for e in rows:
                if e>max_intensity:
                    image[j,i]=max_intensity
                i+=1
            j+=1
        return image


    #predicting the higher resolution image
    def prediction(self):
    """
        Returns the prediction of our input image's super-resolved version
        using the the CNN trained earlier.
        Args:
            no parameters (member function)
        Returns:
            An array of the predicted super-resolved image
        How to be used:
            >> arr_x = np.array(cv.imread("building.png"))#Assume 1020x888 pixels
            >> test = SuperResolution(arr_x)
            >> test.prediction()
            resized (2040, 1776, 1)
            image (2040, 1776, 1)
            cv (2000, 1750, 1)
            35
            patches (1400, 50, 50, 1)
            executed bgr
            (2040, 1776, 3)
            [[[a],[b],...,[z]], [[a],[b],...,[z]], ..., [[a],[b],...,[z]]]
        """

        patches=np.array(sklearn.feature_extraction.image.extract_patches_2d(self.__preprocessing(),(50,50)))

        
        normalized=stats.zscore(np.array(patches).flatten())
        normalized=np.resize(normalized,(patches.shape[0],patches.shape[1],patches.shape[2],1))


        
        predictions=self.network.predict(normalized)
        
        self.predictions=predictions*255
        
        
        
        resized_predictions=np.array(list(map(partial(bicubic_resize,(50,50)),predictions)))       
        
        resized_predictions=np.resize(resized_predictions,tuple(list(resized_predictions.shape)[:3]))
        

        
        result=sklearn.feature_extraction.image.reconstruct_from_patches_2d(resized_predictions,(self.n_y*50,self.n_x*50))
        
       
        
        result=self.__thresholding(result*255,255)
        
        
        
        result=cv.resize(result,(self.shape[1]*self.scale,self.shape[0]*self.scale),interpolation=cv.INTER_CUBIC)

        
        self.output_y=result.astype(np.uint8)
        
        
        self.output=np.dstack((self.output_y,self.output_cr,self.output_cb))
        
        if self.type=="bgr":
            self.output=cv.cvtColor(self.output,cv.COLOR_YCR_CB2BGR)
        elif self.type=='rgb':
            self.output=cv.cvtColor(self.output,cv.COLOR_YCR_CB2RGB)
        
        return self.output
    
    def compare_psnr(self,expected):
        return tf.image.psnr(expected,self.output,255)
                
        


#testing the class
i=70

test=SuperResolution(train_x[i][:100,:100])
test.prediction()

print("Bicubic")
show_bnw(processed_images_x[i][:200,:200])
print("Output")
show_bgrimg(test.output)
print("Grayscale Y")
show_bnw(grayscale_images_y[i][:200,:200])
print("Train_Y")
show_rgbimg(train_y[i][:200,:200])
print("Y")
show_rgbimg(train_y[i][:200,:200])
print("X Bicubic Whole")
show_rgbimg(cv.resize(train_x[i][:100,:100],(200,200),interpolation=cv.INTER_CUBIC))





bicubic=cv.resize(train_x[i],(train_y[i].shape[1],train_y[i].shape[0]),interpolation=cv.INTER_CUBIC)[:200,:200]

print(PSNR(train_y[i][:200,:200],test.output))
print(PSNR(train_y[i][:200,:200],bicubic))



#evaluating
pred_avg=0
bicubic_avg=0
for i in range(20):
    test=SuperResolution(train_x[i][:100,:100])
    test.prediction()
    pred_avg+=PSNR(test.output,train_y[i][:200,:200])
    bicubic_avg+=PSNR(cv.resize(train_x[i],(train_y[i].shape[1],train_y[i].shape[0]),interpolation=cv.INTER_CUBIC)[:200,:200],train_y[i][:200,:200])
    print(i)
print(pred_avg/20)
print(bicubic_avg/20)






