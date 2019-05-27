import numpy as np
import pandas as pd
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import h5py
from keras.models import Sequential
from keras.layers import merge
from convnetskeras.customlayers import splittensor
from keras.layers import concatenate
from sklearn.metrics import confusion_matrix
import itertools
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from keras.callbacks import ModelCheckpoint

K.set_image_data_format('channels_last')

#get_ipython().magic('matplotlib inline')
#get_ipython().magic('matplotlib qt')

#data_x = np.zeros((1000,3,227,227))

scaler = MinMaxScaler()

df = pd.read_csv('dataGrad180All0.csv')

df = df.drop(['Cell Type'],axis=1)


def magGradUx(df):
    mGUx = np.sqrt(np.square(df['grad(Ux):0']) + np.square(df['grad(Ux):1']) + np.square(df['grad(Ux):2']))
    return mGUx

def magGradUy(df):
    mGUy = np.sqrt(np.square(df['grad(Uy):0']) + np.square(df['grad(Uy):1']) + np.square(df['grad(Uy):2']))
    return mGUy

def magGradUz(df):
    mGUz = np.sqrt(np.square(df['grad(Uz):0']) + np.square(df['grad(Uz):1']) + np.square(df['grad(Uz):2']))
    return mGUz

mgux = magGradUx(df)
mgux = mgux.values.reshape(len(mgux),1)
mguy = magGradUy(df)
mguy = mguy.values.reshape(len(mguy),1)
mguz = magGradUz(df)
mguz = mguz.values.reshape(len(mguz),1)

mgux = np.clip(mgux, a_min = 0, a_max = 10000)
mguy = np.clip(mguy, a_min = 0, a_max = 10000)
mguz = np.clip(mguz, a_min = 0, a_max = 10000)

temp1 = np.concatenate((mgux, mguy), axis=1)
temp2 = np.concatenate((temp1, mguz), axis=1)
df2 = np.concatenate(( df['Ux'].values.reshape(len(df['Ux']),1) , df['Uy'].values.reshape(len(df['Uy']),1) , df['Uz'].values.reshape(len(df['Uz']),1) , temp2,  df['nut'].values.reshape(len(df['nut']),1)  ), axis=1)

df2 = pd.DataFrame(df2)
df2 = pd.DataFrame(    np.row_stack([df2.columns, df2.values]),     columns=['Ux','Uy','Uz','gradMagUx', 'gradMagUy', 'gradMagUz','nut'] )

df = df2

index1 = df.index.values
df = df.sample(frac=1)
index2 = df.index.values

dimAll = np.shape(df);

dim2 = dimAll[1];

# 60 %
#df_train = df[:int(np.floor(0.6*len(df)))]
df_train = df
# 20%
rest = df[int(np.floor(0.6*len(df))):]
df_val = rest[:int(np.floor(0.5*len(rest)))]

X_train = df_train.drop(['nut'],axis=1).as_matrix()
#X_train = scaler.fit_transform(df_train.drop(['nut'],axis=1).as_matrix())
y = scaler.fit_transform(df_train['nut'].as_matrix().reshape(-1, 1))

poly = PolynomialFeatures(2,include_bias=False)  # 6 to 27
#poly = PolynomialFeatures(4,include_bias=False)

sz = X_train.shape[0]  #200000  #this is the safe limit for this machine  #df.shape[0]

X2 = poly.fit_transform(X_train[0:sz,:])

dimCNN = X2.shape[1]

A = X2.reshape(( X2.shape[0], X2.shape[1] , 1 , 1  ))
B = X2.reshape(( X2.shape[0], 1, X2.shape[1] , 1  ))

sh = X2.shape[1]
C = np.matmul(A[:,0:sh,0:sh,0],B[:,0:sh,0:sh,0])

for i in range(C.shape[1]):
    C[:, i, :] = scaler.fit_transform(C[:, i, :]) 
    C[:, :, i] = scaler.fit_transform(C[:, :, i]) 


sh = C.shape[1]
C = C.reshape(( X2.shape[0], sh,  sh ,  1   ))

x_train0 = np.zeros((sz,dimCNN,dimCNN,1))
x_train0[:,:,:,0:1] = C

y_train0 = y[0:sz]

img_rows, img_cols = dimCNN, dimCNN
batch_size = 100
epochs = 100

lim = int(0.9*sz)
x_train = x_train0[0:lim,:,:]
x_test = x_train0[lim:sz,:,:]

y_train = y_train0[0:lim,:]
y_test = y_train0[lim:sz,:]

tr_sz = lim
ts_sz = sz - lim

x_train = x_train.reshape(tr_sz,img_rows,img_cols,1)
x_test =  x_test.reshape(ts_sz,img_rows,img_cols,1)


def OficeCNN(weights_path=None):

    inputs = Input(shape=(dimCNN, dimCNN, 1))

    conv_1 = Conv2D(6, 5, 5, activation='relu', name='conv_1')(inputs)
    
    conv_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv_1)
    conv_2 = Conv2D(16, 5, 5, activation='relu', name='conv_2')(conv_2)

    conv_3 = MaxPooling2D((2, 2), strides=(2, 2))(conv_2)

    dense_1 = Flatten(name='flatten')(conv_3)
    dense_1 = Dense(120, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dense(84, activation='relu', name='dense_2')(dense_1)
    dense_3 = Dense(1, name='dense_3_new')(dense_2)
    prediction = Activation('linear', name='output')(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model

def train():   
#    model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True),metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    model.summary()

    checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.8f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]

    model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test), callbacks=callbacks_list)
#    score = oficeCNN.evaluate(x_test, y_test, verbose=0)

def load_trained_model(weights_path):
    # Load model
    model.load_weights(weights_path)

model = OficeCNN()
#weights_path='None'
weights_path='/home/mateus/Keras/oficeDL_3/Weights-010--0.00000526.hdf5'
load_trained_model(weights_path)
train()

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


def denormalize(df,norm_data):
    df = df['nut'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    return new

dfTest = df

ypred = model.predict(x_test)
ypred = denormalize(dfTest,ypred)
y_test = denormalize(dfTest,y_test)

#plt.plot(np.sort(ypred, axis=0),'r'); plt.plot(np.sort(y_test, axis=0),'b')
