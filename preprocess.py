import numpy as np
import random
import json
import PIL.ImageOps
import os
from keras.preprocessing import image as image_utils

def shuffle(data, label):
    temp = list(zip(data, label))
    random.shuffle(temp)
    return zip(*temp)


# setting path and classes mapping

label_path='/home/baker/Desktop/BDD100K/bdd100k_label/labels/100k/train'
image_path='/home/baker/Desktop/BDD100K/bdd100k/images/100k/train'

classes = { "clear":0, "night":1, "overcast":2, "partly cloudy":3, "rainy":4, "snowy":5, "undefined":6 }



files = os.listdir(label_path)
labels=[]
i=0
# load file.json information
for file in files :
    with open( label_path +'/'+ file , 'r') as reader:
        i+=1
        if i%100==0:
            print (i) 
        jf = json.loads(reader.read())
        labels.append(jf)
        


size=100
train_data=[]
train_label=[]
# we convert all image into array so that we can save them into .npy format
counter=0
dest = '/home/baker/Desktop/BDD100K/bdd_digitalize/'+str(size)+'/7.17'

for label in labels :
    
    counter+=1
    
    img = image_utils.load_img(image_path + "/" + label['name']+'.jpg', target_size=(size, size))  # open an image
    #img = PIL.ImageOps.invert(img)  # inverts it
    img = image_utils.img_to_array(img)  # converts it to array
    train_data.append(img)
    
    
    if label['attributes']['timeofday']=='night':
        train_label.append(1)

    elif label['attributes']['weather']=='foggy':
        train_label.append(2)

    else:
        train_label.append(int(classes[label['attributes']['weather']]))
    
    if (counter%100==0):
        print (counter) 
#    for save on batches 
#    if (counter%10000==0):
#        
#        train_data, train_label = shuffle(train_data, train_label)
#        np.save(dest+"/train_data300_" + str(int((counter/10000))) + ".npy",
#        np.array(train_data))  # model root to save image models(image)
#        
#        np.save(dest+"/train_label300_" + str(int((counter/10000))) + ".npy",
#        np.array(train_label))  # model root to save image models(label))
#
#        train_data = []
#        train_label = []
    
#train_data, train_label = shuffle(train_data, train_label)
np.save(dest + "/train_data.npy", np.array(train_data))  # model root to save image models(image)
np.save(dest + "/train_label.npy", np.array(train_label))  # model root to save image models(label)





"""
# for visuallize image is ok 
from matplotlib import pyplot as plt

label_path='/home/baker/Desktop/BDD100K/bdd100k_label/labels/100k/val'
image_path='/home/baker/Desktop/BDD100K/bdd100k/images/100k/val'

classes = { "clear":0, "foggy":1, "overcast":2, "partly cloudy":3, "rainy":4, "snowy":5, "undefined":6 }


files = os.listdir(label_path)
labels=[]
# load file.json information
for file in files :
    with open( label_path +'/'+ file , 'r') as reader:
        jf = json.loads(reader.read())
        labels.append(jf)


size=100
train_data=[]
train_label=[]

counter=0
dest = '/home/baker/Desktop/BDD100K/bdd_digitalize/'+str(size)

for label in labels[:10] :
    
    counter+=1
    
    img = image_utils.load_img(image_path + "/" + label['name']+'.jpg', target_size=(size, size))  # open an image
    
    # show img
    plt.imshow(img)
    plt.show()
    
    img = PIL.ImageOps.invert(img)  # inverts it
    img = image_utils.img_to_array(img)  # converts it to array
    print (label['attributes']['weather'])
    
    train_data.append(img)
    train_label.append(int(classes[label['attributes']['weather']]))

"""







"""
#for analysis data distribution
#----------------------- 
for k in classes:
    count=0
    for i in range(70000):
#        if (labels[i]['attributes']['timeofday']=='dawn/dusk') & (labels[i]['attributes']['weather']==k):
#            count+=1
        if (labels[i]['attributes']['timeofday']=='night') :
            count+=1
    print (count/70000)
"""









