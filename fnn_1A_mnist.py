import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# Load training and testing data (first column = label, rest = 784 pixels)
train = pd.read_csv("HandWritten MNIST/mnist_train.csv")
test = pd.read_csv("HandWritten MNIST/mnist_test.csv")
print(train.shape, test.shape)

x_train = train.drop(columns=["label"]).values
y_train = train["label"].values
x_test = test.drop(columns=["label"]).values
y_test = test["label"].values

# Show first 10 images
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")
plt.show()

# Normalize pixels to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
num_classes = len(lb.classes_)
print("Classes:", num_classes)

# Define FNN architecture
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(x_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])
model.summary()

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(x_train, y_train, epochs=11, batch_size=128, validation_split=0.1)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot accuracy
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.show()
