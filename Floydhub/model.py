import numpy as np
from keras.models import Model
from keras.models import load_model
import keras.backend as K
K.set_image_data_format('channels_last')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y
'''
def load_dataset():
    
    X_train = np.load('./Data/train_196.npy')
    X_test = np.load('./Data/test_196.npy')
    X_val = np.load('./Data/val_196.npy')
    Y_train = convert_to_one_hot(np.load('./Data/Ltrain_196.npy'), 8)
    Y_test = convert_to_one_hot(np.load('./Data/Ltest_196.npy'), 8)
    Y_val = convert_to_one_hot(np.load('./Data/Lval_196.npy'), 8)
    p1 = np.random.permutation(len(X_train))
    p2 = np.random.permutation(len(X_test))
    p3 = np.random.permutation(len(X_val))
    X_train = X_train[p1, :, :, :]
    X_test = X_test[p2, :, :, :]
    X_val = X_val[p3, :, :, :]
    Y_train = Y_train[p1, :]
    Y_test = Y_test[p2, :]
    Y_val = Y_val[p3, :]
    return X_train, X_test, X_val, Y_train, Y_test, Y_val
'''    
def load_by_name(name_X, name_Y):
    X_train = np.load('./Data/'+name_X+'.npy')
    Y_train = np.load('./Data/'+name_Y+'.npy')
    #Y_train = convert_to_one_hot(np.load('./Data/'+name_Y+'.npy'), 8)
    #p1 = np.random.permutation(len(X_train))
    #X_train = X_train[p1, :, :, :]
    #Y_train = Y_train[p1, :]
    return X_train, Y_train

X_train, Y_train = load_by_name('X_full', 'Y_full')
#X_test, Y_test = load_by_name('X_test_f', 'Y_test_f')
model = load_model('model_augment_4.h5') #model_augment_1.h5 
#history = model.fit(x=X, y = Y, validation_data=(X4, Y4), epochs = 1, batch_size = 128)
model.fit(x=X_train, y = Y_train, validation_split=0.15, epochs = 3, batch_size = 512)

model.save('./output/model_augment_5.h5')
model.save_weights('./output/model_augment_5_weights.h5')

#print(model.evaluate(x=X_test, y=Y_test, batch_size=len(X_test)))



