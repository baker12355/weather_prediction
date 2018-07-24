#author: Samet Kalkan

import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import tools as T

def load_train ():
    print ('loading data..')    
    # -----------------load every weather in a balance size -----------------------
    
    size = 224
    dest = '/home/baker/Desktop/BDD100K/digitalize/'+str(size)+'/val'
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
    
    print ('done')  
    return train_data,train_label,train_label_d


def separate_data(v_data, v_label, v_label_1):
    """separates validation data and label according to class no
        Args:
            v_data: validation data to be split
            v_label: validation label to be split
        Returns:
            an array that stores '[val_data,val_label]' in each index for each class.
    """
    vd = [ [[], [],[] ] for _ in range(3)]
    for i in range(len(v_data)):
        cls = int(v_label[i])
        cls_1= int (v_label_1[i])
        vd[cls][0].append(v_data[i])
        vd[cls][1].append(cls)
        vd[cls][2].append(cls_1)
    for i in range(3):
        vd[i][0] = np.array(vd[i][0])
        vd[i][1] = np.array(vd[i][1])
        vd[i][2] = np.array(vd[i][2])
    return vd



classes=['sunny','cloudy','rainy']
day_night=['daytime','night']


validation_data ,validation_label, validation_label_1 = load_train ()


# each index stores a list which stores validation data and its label according to index no
# vd[0] = [val,lab] for class 0
# vd[1] = [val,lab] for class 1 and so on
vd = separate_data(validation_data, validation_label, validation_label_1)

# number of class
num_classes = 3  # Cloudy,Foggy,Rainy,Snowy,Sunny

# for example if label is 4 converts it [0,0,0,0,1]
validation_label = np_utils.to_categorical(validation_label, num_classes)



# loads trained model and architecture
model = load_model("modelsCNN/size224/7.21_Dual_CNN/dual_0.h5")


# -------------------------------------------------------
y = model.predict(validation_data, verbose=1)
y0 = np.argmax(y[0],axis=1)
#y1 = np.argmax(y[1],axis=1)


acc0 = T.get_accuracy_of_class(T.binary_to_class(validation_label), y0)
#acc1 = T.get_accuracy_of_class(T.binary_to_class(validation_label_1), y1)
print("General Accuracy for Weather Data:", round(acc0,2))
#print("General Accuracy for daytime Data:", round(acc1,2))
print("-----------------------------")


print ('Weather:')
for i in range(len(classes)):
    v_data = vd[i][0]
    v_label = vd[i][1]
    y = model.predict(v_data, verbose=0)
    y = np.argmax(y[0],axis=1)
    acc = T.get_accuracy_of_class(v_label, y)
    print("Accuracy for class " + classes[i] + ": ", round(acc,2))
    print("-----------------------------")


print ('Daytime:')
for i in range(len(day_night)):
    v_data = vd[i][0]
    v_label = vd[i][2]
    y = model.predict(v_data, verbose=0)
    y = np.argmax(y[1],axis=1)
    acc = T.get_accuracy_of_class(v_label, y)
    print("Accuracy for class " + day_night[i] + ": ", round(acc,2))
    print("-----------------------------")









    