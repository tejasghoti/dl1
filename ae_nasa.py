import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras import Model
from tensorflow import keras

# 1. Load
df = pd.read_csv("NASA Turbofan ENgine/NASA TurboFan Engine.csv")
print(df.shape)
print(df.head())

# 2. X and y  (Fault_Label: 0 = normal, 1 = fault)
X = df.drop(columns=["Fault_Label", "unit", "cycle", "RUL"])
y = df["Fault_Label"]

# 3. Split + scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_normal = X_train[y_train == 0]      # train only on normal data

# 4. Autoencoder
n = X_train.shape[1]
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(n,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),        # bottleneck
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(n, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="mae")

history = model.fit(X_train_normal, X_train_normal, epochs=20, batch_size=64, validation_split=0.1)

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.show()

# 5. Threshold = 95th percentile of error on normal training data
train_pred = model.predict(X_train_normal)
train_loss = np.mean(np.abs(train_pred - X_train_normal), axis=1)
threshold = np.percentile(train_loss, 95)
print("Threshold:", threshold)

# 6. Test: error > threshold -> anomaly
test_pred = model.predict(X_test)
test_loss = np.mean(np.abs(test_pred - X_test), axis=1)
y_pred = (test_loss > threshold).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))

plt.hist(test_loss, bins=50)
plt.axvline(threshold, color="r")
plt.show()
