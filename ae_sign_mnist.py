import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras import Model
from tensorflow import keras

# 1. Load  (label + 784 pixels, 28x28 images, values 0-255)
train = pd.read_csv("Sign Language MNIST/sign_mnist_train.csv")
test = pd.read_csv("Sign Language MNIST/sign_mnist_test.csv")
print(train.shape, test.shape)
print(train.head())

# 2. X only (no anomaly label here -> just reconstruct images)
X_train = train.drop(columns=["label"]) / 255.0
X_test = test.drop(columns=["label"]) / 255.0

# 3. Autoencoder
n = X_train.shape[1]          # 784
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(n,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),       # bottleneck
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(n, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="mse")

history = model.fit(X_train, X_train, epochs=10, batch_size=256, validation_data=(X_test, X_test))

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.show()

# 4. Reconstruct test images
pred = model.predict(X_test)
print("Test MSE:", np.mean((pred - X_test.values) ** 2))

# 5. Show original (top) vs reconstructed (bottom)
plt.figure(figsize=(15, 4))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test.values[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.subplot(2, 10, i + 11)
    plt.imshow(pred[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()
