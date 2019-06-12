import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#get_ipython().magic('matplotlib inline')

#get_ipython().magic('matplotlib qt')


neuron = 20;

MaxIter = 100

bsz = 32

scaler = MinMaxScaler()

#df = pd.read_csv('data180cyc12All0.csv')

df = pd.read_csv('dataGrad180All0.csv')

df = df.drop(['Cell Type'],axis=1)

#df = df.drop(['Ux','Uy','Uz'],axis=1)

#index = df.index
#df = df.sample(frac=1)
#index = df.index
#df = df.sample(frac=1).reset_index(drop=True)

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

#mgux = np.clip(mgux, a_min = 0, a_max = 10000)
#mguy = np.clip(mguy, a_min = 0, a_max = 10000)
#mguz = np.clip(mguz, a_min = 0, a_max = 10000)

temp1 = np.concatenate((mgux, mguy), axis=1)
temp2 = np.concatenate((temp1, mguz), axis=1)
df2 = np.concatenate(( df['Ux'].values.reshape(len(df['Ux']),1) , df['Uy'].values.reshape(len(df['Uy']),1) , df['Uz'].values.reshape(len(df['Uz']),1) , temp2,  df['nut'].values.reshape(len(df['nut']),1)  ), axis=1)

df2 = pd.DataFrame(df2)
df2 = pd.DataFrame(    np.row_stack([df2.columns, df2.values]),     columns=['Ux','Uy','Uz','gradMagUx', 'gradMagUy', 'gradMagUz','nut'] )
#df2 = pd.DataFrame(    np.row_stack([df2.columns, df2.values]),     columns=['gradMagUx', 'gradMagUy', 'gradMagUz','nut'] )
df2 = df2.drop([0], axis=0)

df = df2

#index1 = df.index.values
#df = df.sample(frac=1)
#index2 = df.index.values


dimAll = np.shape(df);

dim2 = dimAll[1];

# Dataset limitation
sz =3000000 ;    #500000 ;   #100000    #2000

# Comment for entire dataset
#df = df[0:sz] ;

# 60 %
#df_train = df[:int(np.floor(0.6*len(df)))]
df_train = df
# 20%
rest = df[int(np.floor(0.6*len(df))):]
df_val = rest[:int(np.floor(0.5*len(rest)))]

X_train = scaler.fit_transform(df_train.drop(['nut'],axis=1).as_matrix())
y_train = scaler.fit_transform(df_train['nut'].as_matrix().reshape(-1, 1))

X_val = scaler.fit_transform(df_val.drop(['nut'],axis=1).as_matrix())
y_val = scaler.fit_transform(df_val['nut'].as_matrix().reshape(-1, 1))


# 20%
#dfTest = rest[int(np.floor(0.5*len(rest))):]
dfTest = df

X_test = scaler.fit_transform(dfTest.drop(['nut'],axis=1).as_matrix())
y_test = scaler.fit_transform(dfTest['nut'].as_matrix().reshape(-1, 1))

print(X_train.shape)
print(np.max(y_val),np.max(y_train),np.min(y_val),np.min(y_train))

def denormalize(df,norm_data):
    df = df['nut'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    return new


def create_model():
    model = Sequential()

    # The Input Layer :
    model.add(Dense(neuron, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

    # The Hidden Layers :
    model.add(Dense(neuron, kernel_initializer='normal',activation='relu'))
    model.add(Dense(neuron, kernel_initializer='normal',activation='relu'))
    model.add(Dense(neuron, kernel_initializer='normal',activation='relu'))

#    model.add(Dense(neuron, kernel_initializer='normal',activation='relu',name='denseNew_1'))
#    model.add(Dense(neuron, kernel_initializer='normal',activation='relu',name='denseNew_2'))
#    model.add(Dense(neuron, kernel_initializer='normal',activation='relu',name='denseNew_3'))


    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    return model

def train():
    # Compile the network :
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['mean_squared_error'])
    model.summary()

    checkpoint_name = 'Weights-{epoch:03d}--{loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]


    model.fit(X_train, y_train, epochs=MaxIter, batch_size=bsz, validation_split = 0, callbacks=callbacks_list)

weights_path='/home/mateus/Keras/oficeDL_3/mod1/Weights-097--0.00033.hdf5'
#weights_path='/home/mateus/Keras/oficeDL_3/old_weights/Weights-092--0.00075_from1millionCells.hdf5'
def load_trained_model(weights_path):
    # Load model
    model.load_weights(weights_path, by_name=True)

model = create_model()
load_trained_model(weights_path)
train()


ypred = model.predict(X_test)

ypred = denormalize(dfTest,ypred)

y_test = denormalize(dfTest,y_test)


#plt.plot(np.sort(ypred, axis=0),'r'); plt.plot(np.sort(y_test, axis=0),'b')
