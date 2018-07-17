#author: Samet Kalkan

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
np.random.seed(0)


train_data = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/100/7.17/train_data_i.npy")
train_label = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/100/7.17/train_label_i.npy")


#train_data=train_data[:60000]
#train_label=train_label[:60000]

size = train_data.shape[1]

# normalization
train_data = train_data / 255.0

train_data = train_data.reshape(train_data.shape[0], size, size, 3)

# number of class
num_classes = 7 

# for example if label is 4 converts it [0,0,0,0,1]
train_label = np_utils.to_categorical(train_label, num_classes)


val_data=train_data[60000:]
val_label=train_label[60000:]
train_data=train_data[:60000]
train_label=train_label[:60000]





def mycrossentropy(y_true, y_pred, e=0.1):
    return (1-e)*K.categorical_crossentropy(y_true,y_pred) + e*K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)



model = Sequential()

#convolutional layer with 5x5 32 filters and with relu activation function
#input_shape: shape of the each data
#kernel_size: size of the filter
#strides: default (1,1)
#activation: activation function such as "relu","sigmoid"

model.add(Conv2D(32, kernel_size=(2,2),input_shape=(size, size, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

for i in range(4):
    model.add(Conv2D(32, kernel_size=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, kernel_size=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

# beginning of fully connected neural network.
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
# Add fully connected layer with a softmax activation function
model.add(Dense(num_classes, activation='softmax'))


# Compile neural network
model.compile(loss=mycrossentropy, # self_Cross-entropy
                optimizer='adam', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric


checkpointer = ModelCheckpoint(filepath='modelsCNN/size100/7.17_i_1.h5', verbose=1, save_best_only=True)


# begin train the data
history = model.fit(train_data, # train data
            train_label, # label
            validation_data=[val_data,val_label],
            epochs=40, # Number of epochs
            verbose=1,
            batch_size=256,
            callbacks=[checkpointer]
            )

#model.save("modelsCNN/size100/7.17.h5",overwrite=True)





