import numpy as np
import cv2
from keras.models import load_model
from keras.utils import np_utils
import PIL.ImageOps
from PIL import Image
from keras.preprocessing import image as image_utils

classes = ['clear', 'night', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefind']
model = load_model("WeatherPredictionFromImage-development/modelsCNN/size224/7.18_vgg.h5")
print ('load model')


cap = cv2.VideoCapture('video/BSD2017City_F.mp4')
while(cap.isOpened()):
    try:
        ret, frame = cap.read()
        
        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    
        img = img / 255.0
        
        #frame=img
        
        img = np.reshape(img,(1,224,224,3))
        y = model.predict(img, verbose=0)
        y = np.argmax(y,axis=1)
        
        print (classes[y[0]])
        
        
        cv2.putText(frame, classes[y[0]], (10, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    
    
    
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass
    
cap.release()
cv2.destroyAllWindows()


#validation_data = np.load("/home/baker/Desktop/BDD100K/bdd_digitalize/224/7.18/val/val_data.npy")[:10]

