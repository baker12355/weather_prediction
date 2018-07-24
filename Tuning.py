import numpy as np
from keras.utils import np_utils
from keras.models import load_model
np.random.seed(0)



def load_train ():
    print ('loading data..')    
    # -----------------load every weather in a balance size -----------------------
    
    num_classes = 4
    num_classes2= 2
    size = 224
    dest = '/home/baker/Desktop/BDD100K/digitalize/'+str(size)+'/train'
    #train_clear = np.load(dest + "/clear/" + "/train_data.npy" ) # 37410 
    #train_clear = train_clear[:10000]
    
    train_clear      = np.load (dest + "/clear/"  + "/train_data.npy" ) # 37000
    
    train_clear      = train_clear[:10000]

    train_clear      = train_clear/255.0
    
    t_clear_label    = np.array ([3 for i in range(len(train_clear))])

    
    # -----------------load labels - day or night ---------------------------------
    
    
    d_clear_label    = np.load(dest + "/clear/"  + "/train_label.npy" ) 
    d_clear_label    = d_clear_label[:10000]
    
    # -----------------create new labels clear or not -----------------------------
    pass

    # --------------------vertical concat them &  ---------------------------------

    
    t_clear_label   = np_utils.to_categorical(t_clear_label, num_classes)
    d_clear_label = np_utils.to_categorical(d_clear_label, num_classes2)
    
    print ('done')  
    return train_clear ,t_clear_label,d_clear_label



# --------------------------------load data------------------------------------

train_data , train_label , train_label_d = load_train() # 3948 day 6052 night

a = [[0., 0., 0.]for i in range(len(train_data))]
a = np.reshape(a, (10000,3) )
train_label = np.asarray(a, dtype = np.float32 )


from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 

split = train_test_split(train_data, train_label, train_label_d,test_size=0.2, random_state=42)
(trainX, testX, trainWY, testWY,	trainDY, testDY) = split



model = load_model("modelsCNN/size224/7.21_Dual_CNN/dual_0.h5")

#for layer in model.layers:
#    if (layer.name=='category_output' ) | (layer.name=='conv2d_1' ):
#        layer.trainable = False



losses = {
	"category_output": "categorical_crossentropy",
	"color_output": "categorical_crossentropy",
}
lossWeights = {"category_output": 1.0, "color_output": 1.0}
 
# initialize the optimizer and compile the model
print("[INFO] compiling model...")
model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights,metrics=["accuracy"])


checkpointer = ModelCheckpoint(filepath='modelsCNN/size224/7.21_Dual_CNN/dual_tune.h5', verbose=1, save_best_only=True)


H = model.fit(trainX,	{"category_output": trainWY, "color_output": trainDY},
	validation_data=(testX,		{"category_output": testWY, "color_output": testDY}),
    callbacks = [ checkpointer ] ,   
	epochs=30,
	verbose=1,
    batch_size=64,
    )
 







