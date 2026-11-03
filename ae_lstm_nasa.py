import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow import keras

# 1. Load
df = pd.read_csv("NASA Turbofan ENgine/NASA TurboFan Engine.csv")
print(df.shape)
print(df.head())

# 2. Fill missing values
print(df.isnull().sum())
df = df.ffill()                         # fill gaps with the previous reading

# 3. X and y  (Fault_Label: 0 = healthy, 1 = fault)
X = df.drop(columns=["Fault_Label", "unit", "cycle", "RUL"]).values
y = df["Fault_Label"].values
units = df["unit"].values               # engine number

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 4. Make windows of 10 cycles of the SAME engine (LSTM needs sequences)
T = 10
X_seq, y_seq = [], []
for i in range(len(X) - T):
    if units[i] != units[i + T - 1]:    # skip windows that mix two engines
        continue
    X_seq.append(X[i:i + T])
    y_seq.append(y[i + T - 1])          # label of the last step in the window
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print(X_seq.shape)                      # (samples, 10, features)

# 5. Split in order (first 80% engines train, last 20% test)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

X_train_normal = X_train[y_train == 0]  # train only on normal windows

# 6. LSTM autoencoder
n = X_train.shape[2]
model = keras.Sequential([
    keras.layers.LSTM(32, activation="relu", input_shape=(T, n)),   # encoder
    keras.layers.RepeatVector(T),                                    # repeat for each time step
    keras.layers.LSTM(32, activation="relu", return_sequences=True), # decoder
    keras.layers.TimeDistributed(keras.layers.Dense(n))              # rebuild each time step
])
model.compile(optimizer="adam", loss="mae")

history = model.fit(X_train_normal, X_train_normal, epochs=20, batch_size=64, validation_split=0.1)

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.show()

# 7. Threshold = 95th percentile of error on normal training windows
train_pred = model.predict(X_train_normal)
train_loss = np.mean(np.abs(train_pred - X_train_normal), axis=(1, 2))
threshold = np.percentile(train_loss, 95)
print("Threshold:", threshold)

# 8. Test: error > threshold -> anomaly
test_pred = model.predict(X_test)
test_loss = np.mean(np.abs(test_pred - X_test), axis=(1, 2))
y_pred = (test_loss > threshold).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.hist(test_loss, bins=50)
plt.axvline(threshold, color="r")
plt.show()
