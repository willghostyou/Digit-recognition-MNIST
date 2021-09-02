#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras


# In[15]:


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as k
from keras.optimizers import Adadelta,RMSprop
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import load_model


# ## loading data set

# In[3]:


(x_train, y_train),(x_test,y_test) = mnist.load_data()
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))


# ### input image dimensions

# In[4]:


img_rows, img_cols = 28,28


# ## exploring the dataset

# In[5]:


# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()


# ### reshape data to have a single color channel

# In[6]:


if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else :
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# to normalize the pixel values of grayscale images, e.g. rescale them to the range [0,1]. This involves first converting the data type from unsigned integers to floats, then dividing the pixel values by the maximum value.

# In[7]:


x_train= x_train.astype('float32')
x_test= x_test.astype('float32')


# In[8]:


x_train /= 255
x_test /= 255


# In[9]:


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0],'testsamples')


# In[10]:


# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()


# In[11]:


num_classes=10
batch_size=128
epochs=12


# In[ ]:





# ## convert class vectors to binary class matrices

# We also know that there are 10 classes and that classes are represented as unique integers.
# 
# We can, therefore, use a one hot encoding for the class element of each sample, transforming the integer into a 10 element binary vector with a 1 for the index of the class value, and 0 values for all other classes. We can achieve this with the to_categorical() utility function.

# In[12]:


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ## create model

# The model has two main aspects: the feature extraction front end comprised of convolutional and pooling layers, and the classifier backend that will make a prediction.

# In[13]:



model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3,),
            activation= 'relu',
            input_shape= input_shape))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
#dropping out 25% neurons
model.add(Dropout(0.25))
#now we flatten our 28 by 28 by 1 tensor into a vector of 784 elements
model.add(Flatten())
#then we add a fullyconnected layer of 128 neurons
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#fullyconnected layer with 10 output neurons
model.add(Dense(num_classes,activation='softmax'))
model.summary()
#softmax is ideal fit for out onehot encoded vector
#categorical_crossentropy is a very good fit for multiclass classification task
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1,
         validation_data=(x_test, y_test))


# In[18]:


scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='best')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='best')
plt.show()


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("alpha2.h5")


# In[16]:


model = load_model('alpha2.h5')


# In[17]:



model.predict=(x_test[0])


# In[ ]:


# reshape the image
#if k.image_data_format() == 'channels_first':
 #   gray = gray.reshape(gray.shape[0], 1, img_rows, img_cols)
  #  input_shape = (1, img_rows, img_cols)
#else :
 #   gray = gray.reshape(gray.shape[0], img_rows, img_cols, 1)
  #  input_shape = (img_rows, img_cols, 1)
# normalize image
#gray /= 255

# predict digit
#prediction = model.predict(gray)
#print(prediction.argmax())


# In[ ]:


model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model 


# In[ ]:





# In[ ]:




