import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from PIL import Image

data=keras.datasets.mnist

(train_images,train_labels),(test_images,test_labels)=data.load_data()

train_images=train_images/255
test_images=test_images/255


model=keras.Sequential(
    [keras.layers.Flatten(input_shape=(28,28)),
     keras.layers.Dense(128,activation="relu"),
     keras.layers.Dense(10,activation="softmax")])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_images,train_labels,epochs=4)

test_loss,test_acc=model.evaluate(test_images,test_labels)
print(test_acc)

# plt.imshow(train_images[1])
# plt.show()
# plt.xlabel("train")
predicted=model.predict(test_images)
# print(np.argmax(predicted[1]))
# plt.imshow(predicted[1])
# plt.show()

# for i in range(2):
#     plt.grid(False)
#     plt.imshow(test_images[i],cmap=plt.cm.binary)
#     plt.xlabel("Actual: " + str(test_labels[i]))
#     plt.title("Prediction: " + str(np.argmax(predicted[i])))
#
#     plt.show()
image_path = "test2.jpg"  # Change this to the path of your image
user_image = Image.open(image_path).convert('L')  # Convert to grayscale

# Preprocess the user input image
user_image = np.array(user_image.resize((28, 28)))
user_image = user_image / 255.0

# Reshape the image to match the model input shape
user_image = np.reshape(user_image, (1, 28, 28))

# Make a prediction
user_prediction = model.predict(user_image)

# Display the user input and prediction
plt.grid(False)
plt.imshow(np.squeeze(user_image), cmap=plt.cm.binary)
plt.xlabel("User Input")
plt.title("Prediction: " + str(np.argmax(user_prediction)))
plt.show()




