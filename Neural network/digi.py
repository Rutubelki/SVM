import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train.shape
print(x_train[0])

# plt.imshow(x_train[0])
# plt.show()

# plt.imshow(x_train[0],cmap=plt.cm.binary)

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

print(x_train[0])

print(y_train[0])

Img_size=28
x_trainr=np.array(x_train).reshape(-1,Img_size,Img_size,1)
x_testr=np.array(x_test).reshape(-1,Img_size,Img_size,1)
print(x_trainr.shape)
print(x_testr.shape)

print(x_trainr.shape[1:])

model=Sequential()

#First convolution layer
model.add(Conv2D(64,(3,3),input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Second convolution layer
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Third convolution layer
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Fully connected layer 1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

#Fully connected layer 2
model.add(Dense(32))
model.add(Activation("relu"))

#Last Fully connected layer 2
model.add(Dense(10))
model.add(Activation("softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(x_trainr,y_train,epochs=5,validation_split=0.3)