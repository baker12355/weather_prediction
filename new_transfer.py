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



def load_train ():
    print ('loading data..')    
    # -----------------load every weather in a balance size -----------------------
    
    num_classes = 3
    num_classes2= 2
    size = 224
    dest = '/home/baker/Desktop/BDD100K/digitalize/'+str(size)+'/train'
    #train_clear = np.load(dest + "/clear/" + "/train_data.npy" ) # 37410 
    #train_clear = train_clear[:10000]
    
    train_cloudy      = np.load (dest + "/partly cloudy/"  + "/train_data.npy" ) # 4886
    train_overcast    = np.load (dest + "/overcast/"     + "/train_data.npy" )   # 8784
    train_rainy       = np.load (dest + "/rainy/"      + "/train_data.npy" )     # 5070
    
    train_cloudy      = train_cloudy/255.0
    train_overcast    = train_overcast/255.0
    train_rainy       = train_rainy/255.0
    
    t_cloudy_label    = np.array ([0 for i in range(len(train_cloudy))])
    t_overcast_label  = np.array ([1 for i in range(len(train_overcast))])
    t_rainy_label     = np.array ([2 for i in range(len(train_rainy))])
    
    # -----------------load labels - day or night ---------------------------------
    
    d_cloudy_label    = np.load(dest + "/partly cloudy/"  + "/train_label.npy" ) 
    d_overcast_label  = np.load(dest + "/overcast/"   + "/train_label.npy" )    
    d_rainy_label     = np.load(dest + "/rainy/"  + "/train_label.npy" ) 
    
    # -----------------create new labels clear or not -----------------------------
    pass

    # --------------------vertical concat them &  ---------------------------------
    
    train_data    = np.vstack([train_cloudy,train_overcast,train_rainy])
    train_label   = np.hstack([t_cloudy_label,t_overcast_label,t_rainy_label])
    train_label_d = np.hstack([d_cloudy_label,d_overcast_label,d_rainy_label])
    
    
    train_label   = np_utils.to_categorical(train_label, num_classes)
    train_label_d = np_utils.to_categorical(train_label_d, num_classes2)
    print ('done')  
    return train_data,train_label,train_label_d



# --------------------------------load data------------------------------------
    
train_data , train_label , train_label_d = load_train()


num_classes_0 = 3
num_classes_1 = 2

from sklearn.model_selection import train_test_split

split = train_test_split(train_data, train_label, train_label_d,test_size=0.2, random_state=42)
(trainX, testX, trainWY, testWY,	trainDY, testDY) = split

from keras import backend as K

def mycrossentropy(y_true, y_pred, e=0.1):
    return (1-e)*K.categorical_crossentropy(y_true,y_pred) + e*K.categorical_crossentropy(K.ones_like(y_pred)/num_classes_0, y_pred)


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


# --------------------- batch -------------------------

from resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.models import Model

model = ResNet50(weights='imagenet',include_top=False)
#model.summary()
last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(128, activation='relu',name='fc-1')(x)
x = Dense(128, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
out = Dense(num_classes_0, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)

#custom_resnet_model2.summary()
#
#for layer in custom_resnet_model2.layers[:-6]:
#	layer.trainable = False

custom_resnet_model2.compile(loss= mycrossentropy ,optimizer='adam',metrics=['accuracy'])
custom_resnet_model2.summary()

checkpointer = ModelCheckpoint(filepath='modelsCNN/size224/7.21_transfer/7.21_res_0.h5', verbose=1, save_best_only=True)


# ---------------res ------------------

hist = custom_resnet_model2.fit(train_data, train_label, 
                             shuffle=True,
                             batch_size=50, 
                             epochs=10, 
                             verbose=1, 
                             validation_data=(testX, testWY),
                             callbacks=[checkpointer])





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
#for layer in custom_vgg_model2.layers[:-3]:
#	layer.trainable = False

custom_vgg_model2.summary()

custom_vgg_model2.compile(loss=mycrossentropy,optimizer='adam',metrics=['accuracy'])




checkpointer = ModelCheckpoint(filepath='modelsCNN/size224/7.21_transfer/7.21_vgg_0.h5', verbose=0, save_best_only=True)

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



# ---------------vgg ------------------
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










