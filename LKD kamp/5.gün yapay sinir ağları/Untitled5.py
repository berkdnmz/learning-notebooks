#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


# In[38]:


X_train = np.loadtxt("./data/input.csv", delimiter = ",")
Y_train = np.loadtxt("./data/labels.csv", delimiter = ",")

X_test = np.loadtxt("./data/input_test.csv", delimiter = ",")
Y_test = np.loadtxt("./data/labels_test.csv", delimiter = ",")


# In[39]:


X_train = X_train.reshape(len(X_train), 100, 100,3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

X_train = X_train / 255.0
X_test = X_test / 255.0


# In[40]:


idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx, :])
plt.show()


# In[41]:


model = Sequential([
    Conv2D(32, (3,3), activation = "relu", input_shape = (100,100,3)),
    MaxPool2D((2,2)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPool2D((2,2)),

    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model = Sequential()

model.add(Conv2D(32, (3,3), activation= "relu", input_shape = (100, 100, 3)))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(32, (3,3), activation= "relu", ))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))


# In[42]:


model.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[43]:


model.fit(X_train, Y_train, epochs=10, batch_size=64)


# In[44]:


model.evaluate(X_test, Y_test)


# In[55]:


idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()

y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

if(y_pred == 0):
    pred = "dog"
else:
    pred = "cat"
print(f"Our model says it is a : {pred}")

