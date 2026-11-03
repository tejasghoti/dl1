import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data (one file, last column = label, 3072 pixels = 32x32x3) -> split 80/20
df = pd.read_csv("CIFAR10/cifar-10.csv")
print(df.shape)

x = df.drop(columns=["label"]).values
y = df["label"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Show first 10 images (csv stores all red, then green, then blue -> rebuild image)
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(3, 32, 32).transpose(1, 2, 0))
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

# Reshape for CNN
x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Define CNN architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=x_train.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

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
