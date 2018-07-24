from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2
import sys
import os 
from keras.models import load_model

classes = [ 'sunny','overcast',  'rainy']
classes_d = ['day','night']

model1 = load_model('/home/baker/Desktop/WeatherPredictionFromImage-development/modelsCNN/size224/7.21_Dual_CNN/dual_tune.h5')

n=0
def heapmap(img):
    global n 
    n+=1
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    
    preds = model1.predict(x)
    class_idx = np.argmax(preds[1])
    
    class_output = model1.output[1][:, class_idx]
    last_conv_layer = model1.get_layer("conv2d_6")
    
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model1.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    for i in range(32):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    cv2.putText(img, classes_d[class_idx], (10, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (0, 255, 255), 5, cv2.LINE_AA)
    
    img = cv2.resize(img, (500, 500)) 
    superimposed_img = cv2.resize(superimposed_img, (500, 500)) 
    
    numpy_horizontal = np.hstack((img, superimposed_img))
    
    cv2.imwrite('/home/baker/Desktop/BDD100K/visualize_d/'+str(n)+'.png',numpy_horizontal)
    
    
#    cv2.imshow("Original", numpy_horizontal)
#    cv2.moveWindow("Original", 40,30)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


path = '/home/baker/Desktop/BDD100K/bdd100k/images/100k/val/'
files = os.listdir(path)

for file in files[:100]:
    img_path = path + file
    img = image.load_img (img_path, target_size=(224, 224))
    heapmap(img)
    






