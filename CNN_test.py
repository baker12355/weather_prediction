#author: Samet Kalkan

import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import tools as T


#validation_data = np.load("/home/baker/Desktop/Image2Weather dataset/image_digitalize100/size100train_data.npy")
#validation_label = np.load("/home/baker/Desktop/Image2Weather dataset/image_digitalize100/size100train_label.npy")

validation_data = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/100/7.17/train_data_i.npy")
validation_label = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/100/7.17/train_label_i.npy")


validation_data=validation_data[60000:]
validation_label=validation_label[60000:]

# normalization
validation_data = validation_data / 255.0



# each index stores a list which stores validation data and its label according to index no
# vd[0] = [val,lab] for class 0
# vd[1] = [val,lab] for class 1 and so on
vd = T.separate_data(validation_data, validation_label)

# number of class
num_classes = 7  # Cloudy,Foggy,Rainy,Snowy,Sunny

# for example if label is 4 converts it [0,0,0,0,1]
validation_label = np_utils.to_categorical(validation_label, num_classes)


# loads trained model and architecture
model = load_model("modelsCNN/size100/7.17_i_1.h5")


# -------predicting part-------
y = model.predict_classes(validation_data, verbose=1)
acc = T.get_accuracy_of_class(T.binary_to_class(validation_label), y)
print("General Accuracy for Validation Data:", acc)
print("-----------------------------")


for i in range(len(vd)):
    v_data = vd[i][0]
    v_label = vd[i][1]
    y = model.predict_classes(v_data, verbose=0)
    acc = T.get_accuracy_of_class(v_label, y)
    print("Accuracy for class " + T.classes[i] + ": ", acc)
    print("-----------------------------")
    
    