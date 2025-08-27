
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



#Xor veri seti
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()

model.add(Dense(2, input_dim=2, activation="sigmoid")) # gizli katman

model.add(Dense(1, activation="sigmoid"))


model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


model.fit(X, y, epochs = 5000, verbose = 0)

pred = model.predict(X)

print(pred.round())





