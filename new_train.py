import numpy as np 
from keras.utils import np_utils


classes = { "clear":0, "foggy":1, "overcast":2, "partly cloudy":3,
           "rainy":4, "snowy":5, "undefined":6 }

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



# --------------------------------build model ---------------------------------

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.callbacks import ModelCheckpoint , EarlyStopping

class FashionNet:
	@staticmethod
	def build_category_branch(inputs, numCategories,
		finalAct="softmax", chanDim=-1):
		# utilize a lambda layer to convert the 3 channel input to a
		# grayscale representation
		#x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)
 
		# CONV => RELU => POOL
		x = Conv2D(32, (3, 3), padding="same",name='W01')(inputs)
		x = Activation("relu",name='W02')(x)
		x = BatchNormalization(axis=chanDim,name='W03')(x)
		x = MaxPooling2D(pool_size=(3, 3),name='W04')(x)
		x = Dropout(0.25,name='W05')(x)
        
        #CONV => RELU) * 2 => POOL
		x = Conv2D(64, (3, 3), padding="same",name='W06')(x)
		x = Activation("relu",name='W07')(x)
		x = BatchNormalization(axis=chanDim,name='W08')(x)
		x = Conv2D(64, (3, 3), padding="same",name='W09')(x)
		x = Activation("relu",name='W10')(x)
		x = BatchNormalization(axis=chanDim,name='W11')(x)
		x = MaxPooling2D(pool_size=(2, 2),name='W12')(x)
		x = Dropout(0.25,name='W13')(x)
 
		# (CONV => RELU) * 2 => POOL
		x = Conv2D(128, (3, 3), padding="same",name='W14')(x)
		x = Activation("relu",name='W15')(x)
		x = BatchNormalization(axis=chanDim,name='W16')(x)
		x = Conv2D(128, (3, 3), padding="same",name='W17')(x)
		x = Activation("relu",name='W18')(x)
		x = BatchNormalization(axis=chanDim,name='W19')(x)
		x = MaxPooling2D(pool_size=(2, 2),name='W20')(x)
		x = Dropout(0.25)(x)
        
        
		# (CONV => RELU) * 2 => POOL
		x = Conv2D(256, (3, 3), padding="same",name='W21')(x)
		x = Activation("relu",name='W22')(x)
		x = BatchNormalization(axis=chanDim,name='W23')(x)
		x = Conv2D(256, (3, 3), padding="same",name='W24')(x)
		x = Activation("relu",name='W25')(x)
		x = BatchNormalization(axis=chanDim,name='W26')(x)
		x = MaxPooling2D(pool_size=(2, 2),name='W27')(x)
		x = Dropout(0.25,name='W28')(x)
        
        
        
        
		# define a branch of output layers for the number of different
		# clothing categories (i.e., shirts, jeans, dresses, etc.)
		x = Flatten(name='W29')(x)
		x = Dense(512,name='W30')(x)
		x = Activation("relu",name='W31')(x)
		x = BatchNormalization(name='W32')(x)
		x = Dropout(0.5,name='W33')(x)
		x = Dense(numCategories,name='W34')(x)
		x = Activation(finalAct, name="category_output")(x)
 
		# return the category prediction sub-network
		return x

	@staticmethod
	def build_color_branch(inputs, numColors, finalAct="softmax",
		chanDim=-1):
		# CONV => RELU => POOL
		x = Conv2D(16, (3, 3), padding="same")(inputs)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(3, 3))(x)
		x = Dropout(0.25)(x)
 
		# CONV => RELU => POOL
		x = Conv2D(32, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)
 
		# CONV => RELU => POOL
		x = Conv2D(32, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)
		# define a branch of output layers for the number of different
		# colors (i.e., red, black, blue, etc.)
		x = Flatten()(x)
		x = Dense(128)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(numColors)(x)
		x = Activation(finalAct, name="color_output")(x)
 
		# return the color prediction sub-network
		return x
        
        
	@staticmethod
	def build(width, height, numCategories, numColors,
		finalAct="softmax"):
		# initialize the input shape and channel dimension (this code
		# assumes you are using TensorFlow which utilizes channels
		# last ordering)
		inputShape = (height, width, 3)
		chanDim = -1
 
		# construct both the "category" and "color" sub-networks
		inputs = Input(shape=inputShape)
		categoryBranch = FashionNet.build_category_branch(inputs,
			numCategories, finalAct=finalAct, chanDim=chanDim)
		colorBranch = FashionNet.build_color_branch(inputs,
			numColors, finalAct=finalAct, chanDim=chanDim)
 
		# create the model using our input (the batch of images) and
		# two separate outputs -- one for the clothing category
		# branch and another for the color branch, respectively
		model = Model(
			inputs=inputs,
			outputs=[categoryBranch, colorBranch],
			name="fashionnet")
 
		# return the constructed network architecture
		return model


from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

split = train_test_split(train_data, train_label, train_label_d,test_size=0.2, random_state=42)
(trainX, testX, trainWY, testWY,	trainDY, testDY) = split


# initialize our FashionNet multi-output network

EPOCHS = 15
INIT_LR = 1e-3
BS = 256

a = FashionNet()
model = a.build(224 , 224 ,3 , 2 ,finalAct="softmax")


# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"category_output": "categorical_crossentropy",
	"color_output": "categorical_crossentropy",
}
lossWeights = {"category_output": 1.0, "color_output": 0.01}
 
# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])



earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

checkpointer = ModelCheckpoint(filepath='modelsCNN/size224/7.21_Dual_CNN/dual_0.h5', verbose=1, save_best_only=True)


H = model.fit(trainX,	{"category_output": trainWY, "color_output": trainDY},
	validation_data=(testX,		{"category_output": testWY, "color_output": testDY}),
    callbacks = [ checkpointer ] ,   
	epochs=EPOCHS,
	verbose=1,
    batch_size=64,
    )
 








