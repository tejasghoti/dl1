import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# a. Set up: dataset folder (one sub-folder per class) and image size
DATA_DIR = "Intel Image Classification"
IMG_SIZE = (128, 128)

# b. Load data (80% train, 20% test) -- labels come from folder names
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=32)
test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=32)

class_names = train_ds.class_names
print("Classes:", len(class_names), class_names)

# c. Pre-trained model: VGG16 without its top (classifier) layers
base = VGG16(weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
             include_top=False, input_shape=IMG_SIZE + (3,))

# d. Freeze base layers + add our own layers on top
base.trainable = False

model = Sequential([
    Lambda(preprocess_input),             # VGG16 preprocessing
    base,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# e. Train
history = model.fit(train_ds, validation_data=test_ds, epochs=5)

# f. Loss curve
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend()
plt.show()

# g. Evaluate
loss, accuracy = model.evaluate(test_ds)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# h. Visualize predictions (9 test images)
for images, labels in test_ds.take(1):
    preds = np.argmax(model.predict(images), axis=1)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title("True: " + class_names[labels[i]] + "\nPred: " + class_names[preds[i]])
        plt.axis("off")
    plt.show()

# i. Fine tune: unfreeze last 4 layers of VGG16, train again with small learning rate
base.trainable = True
for layer in base.layers[:-4]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=test_ds, epochs=3)

loss, accuracy = model.evaluate(test_ds)
print("After Fine Tuning - Test Loss:", loss)
print("After Fine Tuning - Test Accuracy:", accuracy)
