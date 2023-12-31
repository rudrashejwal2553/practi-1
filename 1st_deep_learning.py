# -*- coding: utf-8 -*-
"""1st deep learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16JMiZng4ZrL4lBeNDnI2mWPa_HHTrRNg
"""

import numpy as np

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape

x_test.shape

import matplotlib.pyplot as plt

x = np.zeros(100).reshape(10,10)

x

plt.imshow(x, cmap = 'gray')

plt.imshow(x_train[220] ,cmap = 'gray')

y_train[100]

plt.imshow(x_train[200] ,cmap = 'gray')
y_train[200]

x = np.array([[2,3,5],[8,9,0]])

x

x.shape

img  =x_train[3]
img.shape

img = img.flatten()

img.shape

x_test.shape

x_train = x_train.reshape(60000,784)
x_test =  x_test.reshape(10000,784)

