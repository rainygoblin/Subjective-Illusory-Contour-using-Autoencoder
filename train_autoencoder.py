# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:21:48 2018

@author: user
"""

from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cv2

from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D  #Dropout
from keras import optimizers
from keras.utils import plot_model

def autoencoder(input_size = (64,64,1), depth = 3):
     
    inputs = Input(input_size)
    current_layer = inputs #forループ中でレイヤ名を保存する変数
 
    # add levels with max pooling
    for layer_depth in range(depth):
        n_filters = 64*(2**layer_depth)
        layer1 = Conv2D(n_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(current_layer)
        layer2 = Conv2D(n_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer1)
 
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=(2,2))(layer2)
        else:
            current_layer = layer2
    
    # add levels with up-sampling
    for layer_depth in range(depth-2, -1, -1):   #depth-2から0まで-1ずつ減る
        n_filters = 64*(2**layer_depth)
        up = Conv2D(n_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(current_layer))
        current_layer = Conv2D(n_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up)
        current_layer = Conv2D(n_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(current_layer)
         
    current_layer = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(current_layer)
    output = Conv2D(1, 1, activation='sigmoid')(current_layer)
    model = Model(input = inputs, output = output)
 
    model.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy')
     
    model.summary()
 
    return model

def load_data():
    #preprocessing
    # フォルダの中にある画像を順次読み込む
    Kanizsa_X = np.zeros((27200, 64, 64, 1))
    Kanizsa_Y = np.zeros((27200, 64, 64, 1))
    Ehrenstein_X = np.zeros((9375, 64, 64, 1))
    Ehrenstein_Y = np.zeros((9375, 64, 64, 1))

    # inputの画像
    for n_img in range(27200):
        img = cv2.imread("./data/Kanizsa_X/Kanizsa_X_"+"{0:05d}".format(n_img)+".png", 
                         cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (-1, 64, 64, 1))
        Kanizsa_X[n_img] = img
        
    for n_img in range(9375):
        img = cv2.imread("./data/Ehrenstein_X/Ehrenstein_X_"+"{0:05d}".format(n_img)+".png",
                         cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (-1, 64, 64, 1))
        Ehrenstein_X[n_img] = img 
        
    # outputの画像
    for n_img in range(27200):
        img = cv2.imread("./data/Kanizsa_Y/Kanizsa_Y_"+"{0:05d}".format(n_img)+".png",
                         cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (-1, 64, 64, 1))
        Kanizsa_Y[n_img] = img 
        
    for n_img in range(9375):
        img = cv2.imread("./data/Ehrenstein_Y/Ehrenstein_Y_"+"{0:05d}".format(n_img)+".png",
                         cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (-1, 64, 64, 1))
        Ehrenstein_Y[n_img] = img 

    return Kanizsa_X, Kanizsa_Y, Ehrenstein_X, Ehrenstein_Y
    
def load_train_data():
    #preprocessing
    # フォルダの中にある画像を順次読み込む
    x_train = np.zeros((30000, 64, 64, 1))
    y_train = np.zeros((30000, 64, 64, 1))

    # inputの画像
    for n_img in range(30000):
        img = cv2.imread("./data/Kanizsa_X/Kanizsa_randsq_X_"+"{0:05d}".format(n_img)+".png", 
                         cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (-1, 64, 64, 1))
        x_train[n_img] = img
        
        
    # outputの画像
    for n_img in range(30000):
        img = cv2.imread("./data/Kanizsa_Y/Kanizsa_randsq_Y_"+"{0:05d}".format(n_img)+".png",
                         cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (-1, 64, 64, 1))
        y_train[n_img] = img 

    return x_train, y_train
    
#def train():
    #20400, 9375

Kanizsa_X, Kanizsa_Y, Ehrenstein_X, Ehrenstein_Y = load_data()
    
#x_train = np.concatenate((Kanizsa_X[:20000], Ehrenstein_X[:7000]), axis=0)
#y_train = np.concatenate((Kanizsa_Y[:20000], Ehrenstein_Y[:7000]), axis=0)
#x_test = Kanizsa_X
#y_test = Kanizsa_Y
x_test = Ehrenstein_X
y_test = Ehrenstein_Y
#x_test = np.concatenate((Kanizsa_X[20000:], Ehrenstein_X[7000:]), axis=0)
#y_test = np.concatenate((Kanizsa_Y[20000:], Ehrenstein_Y[7000:]), axis=0)

x_train, y_train = load_train_data()

x_train = x_train.astype('float32') / 255
y_train = y_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_test = y_test.astype('float32') / 255

model = autoencoder()

#model.load_weights('illusory_u3k_best_model_weights.h5')
"""
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

hist = model.fit(x_train, y_train,
                 validation_split=0.2,
                 epochs=500,
                 batch_size=128,
                 callbacks=[early_stopping])

model.save_weights('illusory_VE_model_weights.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='loss')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('AEtrainCurve.png')
"""
model.load_weights('illusory_VE_model_weights.h5')

# Predict the Autoencoder output from corrupted test images
y_predict = model.predict(x_test)
y_subtraction = 1-np.abs(x_test - y_predict)
# Display the 1st 8 corrupted and denoised images
rows, cols = 1, 100
num = rows * cols
num_start = 5000
imgs = np.concatenate([x_test[num_start:num_start+num], y_test[num_start:num_start+num],
                       y_predict[num_start:num_start+num], y_subtraction[num_start:num_start+num]])
imgs = imgs.reshape((rows * 4, cols, 64, 64))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 4, -1, 64, 64))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('AE_result3.png')
plt.show()
"""
# Predict the Autoencoder output from corrupted test images
y_predict = model.predict(x_train)
y_subtraction = 1 - np.abs(x_train - y_predict)
# Display the 1st 8 corrupted and denoised images
rows, cols = 1, 20
num = rows * cols
num_start = 0
imgs = np.concatenate([x_train[num_start:num_start+num], y_train[num_start:num_start+num],
                       y_predict[num_start:num_start+num], y_subtraction[num_start:num_start+num]])
imgs = imgs.reshape((rows * 4, cols, 64, 64))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 4, -1, 64, 64))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)

plt.figure()
plt.axis('off')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('AE_result_inverse.png')
plt.show()
"""