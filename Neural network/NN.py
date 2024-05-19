import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#LOADING THE DATA
data=keras.datasets.fashion_mnist

class_names=["T-shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

(train_images,train_labels),(test_images,test_labels)=data.load_data()
train_images=train_images/245
test_images=test_images/245

# print(train_images[5])
# plt.imshow(train_images[5],cmap=plt.cm.binary)
# plt.show()

#TRAINING THE MODEL
model=keras.Sequential(
    [keras.layers.Flatten(input_shape=(28,28)),
     keras.layers.Dense(128,activation="relu"),
     keras.layers.Dense(10,activation="softmax")
     ])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_images,train_labels,epochs=5)

# test_loss,test_acc=model.evaluate(test_images,test_labels)
# print("Test_accuracy=",test_acc)

prediction=model.predict(test_images)
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel("Actual:"+class_names[test_labels[i]])
    plt.title("Prediction:"+[np.argmax(prediction[i])])
    plt.show()

