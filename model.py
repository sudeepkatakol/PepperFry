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
    
model = load_model('model_augment_5.h5')

for i in range(2):
    X_train, Y_train = load_by_name('X_full', 'Y_full')
    model.fit(x=X_train, y = Y_train, validation_split=0.08, epochs = 1, batch_size = 256)
    del X_train
    del Y_train
    
    model.save('./output/'+str(i)+'.h5')
    model.save_weights('./output/'+str(i)+'.h5')

X_test_g = np.load('./Data/X_test_google.npy')
predictions = model.predict(X_test_g)
del X_test_g
'''
X_test, Y_test = load_by_name('X_test_f', 'Y_test_f')
print(model.evaluate(x=X_test, y=Y_test, batch_size = 512))
del X_test
del Y_test
'''
X_google_test, Y_google_test = load_by_name('X_google_test', 'Y_google_test')
print()
print(model.evaluate(x=X_google_test, y=Y_google_test, batch_size = 512))
print()
print("Google predictions: ")
print(np.argmax(predictions, 1))
'''

