#author: Samet Kalkan

import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.layers import Input
from resnet50 import ResNet50
from vgg16 import VGG16
import time
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.models import Model
np.random.seed(0)

def load_training(order=1):
    file1 = str(order)
    file2 = str(order+1)
    
    train_data = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/224/7.18/train_data_224_"+file1+".npy")
    train_label = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/224/7.18/train_label_224_"+file1+".npy")
    
    train_data1 = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/224/7.18/train_data_224_"+file2+".npy")
    train_label1 = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/224/7.18/train_label_224_"+file2+".npy")
    
    train_data = np.vstack([train_data,train_data1])
    train_label= np.hstack([train_label,train_label1])
    del train_data1,train_label1
    
    size = train_data.shape[1]
    # normalization
    train_data = train_data / 255.0
    train_data = train_data.reshape(train_data.shape[0], size, size, 3)
    train_label = np_utils.to_categorical(train_label, num_classes)
    print ('training data load !')
    return train_data,train_label


def load_val():
    val_data = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/224/7.18/val/val_data.npy")
    val_label = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/224/7.18/val/val_label.npy")
    val_data=val_data[8000:]
    val_label=val_label[8000:]
    val_data = val_data / 255.0
    size = val_data.shape[1]
    val_data = val_data.reshape(val_data.shape[0], size, size, 3)
    val_label = np_utils.to_categorical(val_label, num_classes)
    print ('validation data load !')
    return val_data,val_label


num_classes = 7 
#train_data,train_label=load_training()
#val_data,val_label=load_val()

# number of class


def mycrossentropy(y_true, y_pred, e=0.1):
    return (1-e)*K.categorical_crossentropy(y_true,y_pred) + e*K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)


"""
model = ResNet50(weights='imagenet',include_top=False)
#model.summary()
last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(128, activation='relu',name='fc-1')(x)
x = Dense(128, activation='relu',name='fc-2')(x)
# a softmax layer for 4 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)

#custom_resnet_model2.summary()

#for layer in custom_resnet_model2.layers[:-6]:
#	layer.trainable = False

custom_resnet_model2.compile(loss= mycrossentropy ,optimizer='adam',metrics=['accuracy'])
custom_resnet_model2.summary()
"""

image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()

last_layer = model.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
custom_vgg_model2.summary()

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

custom_vgg_model2.summary()

custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])




checkpointer = ModelCheckpoint(filepath='modelsCNN/size100/7.18_vgg_0.h5', verbose=0, save_best_only=True)

print ('model load !')


#t=time.time()
#hist = custom_resnet_model2.fit(train_data, train_label, 
#                               batch_size=50, 
#                               epochs=12, 
#                               verbose=1, 
#                               validation_data=(val_data, val_label),
#                               callbacks=[checkpointer])
#
#print('Training time: %s' % ( time.time())- t)
#
#(loss, accuracy) = custom_resnet_model2.evaluate(val_data, val_label, batch_size=100, verbose=1)
#
#print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
#


# --------------------- batch -------------------------

val_data,val_label=load_val()

for i in range(1,7,2):
    print (i , 'th')
    train_data , train_label = load_training(i)
    
    hist = custom_vgg_model2.fit(train_data, train_label, 
                                 shuffle=True,
                                 batch_size=50, 
                                 epochs=20, 
                                 verbose=1, 
                                 validation_data=(val_data, val_label),
                                 callbacks=[checkpointer])





















