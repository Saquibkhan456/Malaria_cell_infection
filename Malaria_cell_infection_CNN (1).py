#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
# Technically not necessary in newest versions of jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


my_data_dir = 'C:\\Users\\Marcial\\Pierian-Data-Courses\\cell_images'


# In[5]:


# CONFIRM THAT THIS REPORTS BACK 'test', and 'train'
os.listdir(my_data_dir) 


# In[6]:


test_path = my_data_dir+'\\test\\'
train_path = my_data_dir+'\\train\\'


# In[7]:


os.listdir(test_path)


# In[8]:


os.listdir(train_path)


# In[11]:


os.listdir(train_path+'\\parasitized')[0]


# In[50]:


para_cell = train_path+'\\parasitized'+'\\C100P61ThinF_IMG_20150918_144104_cell_162.png'


# In[51]:


para_img= imread(para_cell)


# In[52]:


plt.imshow(para_img)


# In[53]:


para_img.shape


# In[54]:


unifected_cell_path = train_path+'\\uninfected\\'+os.listdir(train_path+'\\uninfected')[0]
unifected_cell = imread(unifected_cell_path)
plt.imshow(unifected_cell)


# **Let's check how many images there are.**

# In[55]:


len(os.listdir(train_path+'\\parasitized'))


# In[56]:


len(os.listdir(train_path+'\\uninfected'))


# **Let's find out the average dimensions of these images.**

# In[57]:


unifected_cell.shape


# In[58]:


para_img.shape


# In[36]:


# Other options: https://stackoverflow.com/questions/1507084/how-to-check-dimensions-of-all-images-in-a-directory-using-python
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'\\uninfected'):
    
    img = imread(test_path+'\\uninfected'+'\\'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)


# In[38]:


sns.jointplot(dim1,dim2)


# In[40]:


np.mean(dim1)


# In[41]:


np.mean(dim2)


# In[42]:


image_shape = (130,130,3)


# ## Preparing the Data for the model
# 
# There is too much data for us to read all at once in memory. We can use some built in functions in Keras to automatically process the data, generate a flow of batches from a directory, and also manipulate the images.
# 
# ### Image Manipulation
# 
# Its usually a good idea to manipulate the images with rotation, resizing, and scaling so the model becomes more robust to different images that our data set doesn't have. We can use the **ImageDataGenerator** to do this automatically for us. Check out the documentation for a full list of all the parameters you can use here!

# In[75]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[76]:


help(ImageDataGenerator)


# In[44]:


image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


# In[60]:


plt.imshow(para_img)


# In[63]:


plt.imshow(image_gen.random_transform(para_img))


# In[64]:


plt.imshow(image_gen.random_transform(para_img))


# ### Generating many manipulated images from a directory
# 
# 
# In order to use .flow_from_directory, you must organize the images in sub-directories. This is an absolute requirement, otherwise the method won't work. The directories should only contain images of one class, so one folder per class of images.
# 
# Structure Needed:
# 
# * Image Data Folder
#     * Class 1
#         * 0.jpg
#         * 1.jpg
#         * ...
#     * Class 2
#         * 0.jpg
#         * 1.jpg
#         * ...
#     * ...
#     * Class n

# In[65]:


image_gen.flow_from_directory(train_path)


# In[66]:


image_gen.flow_from_directory(test_path)


# # Creating the Model

# In[67]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D


# In[68]:


#https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, remember its binary so we use sigmoid
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[69]:


model.summary()


# ## Early Stopping

# In[70]:


from tensorflow.keras.callbacks import EarlyStopping


# In[71]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# ## Training the Model

# In[74]:


help(image_gen.flow_from_directory)


# In[77]:


batch_size = 16


# In[78]:


train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')


# In[113]:


test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)


# In[82]:


train_image_gen.class_indices


# In[83]:


import warnings
warnings.filterwarnings('ignore')


# In[84]:


results = model.fit_generator(train_image_gen,epochs=20,
                              validation_data=test_image_gen,
                             callbacks=[early_stop])


# In[88]:


from tensorflow.keras.models import load_model
model.save('malaria_detector.h5')


# # Evaluating the Model

# In[89]:


losses = pd.DataFrame(model.history.history)


# In[90]:


losses[['loss','val_loss']].plot()


# In[91]:


model.metrics_names


# In[114]:


model.evaluate_generator(test_image_gen)


# In[97]:


from tensorflow.keras.preprocessing import image


# In[115]:


# https://datascience.stackexchange.com/questions/13894/how-to-get-predictions-with-predict-generator-on-streaming-test-data-in-keras
pred_probabilities = model.predict_generator(test_image_gen)


# In[116]:


pred_probabilities


# In[117]:


test_image_gen.classes


# In[118]:


predictions = pred_probabilities > 0.5


# In[119]:


# Numpy can treat this as True/False for us
predictions


# In[120]:


from sklearn.metrics import classification_report,confusion_matrix


# In[121]:


print(classification_report(test_image_gen.classes,predictions))


# In[122]:


confusion_matrix(test_image_gen.classes,predictions)


# # Predicting on an Image

# In[124]:


# Your file path will be different!
para_cell


# In[132]:


my_image = image.load_img(para_cell,target_size=image_shape)


# In[137]:


my_image


# In[138]:


type(my_image)


# In[139]:


my_image = image.img_to_array(my_image)


# In[142]:


type(my_image)


# In[143]:


my_image.shape


# In[144]:


my_image = np.expand_dims(my_image, axis=0)


# In[145]:


my_image.shape


# In[146]:


model.predict(my_image)


# In[147]:


train_image_gen.class_indices


# In[148]:


test_image_gen.class_indices

