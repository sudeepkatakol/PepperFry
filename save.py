import numpy as np
import cv2
import sys
import os
from os import listdir
from os.path import isfile, join
mypath = "/home/sudeep/Desktop/Deep Learning/Pepperfry/Google_Furniture"
classes = next(os.walk(mypath))[1]
files = []
for x in classes:
    path = join(mypath, x)
    files.append([path + "/" + f for f in listdir(path) if isfile(join(path, f)) and f[0] != '.'])
flatten = lambda l: [item for sublist in l for item in sublist]
for i in range(len(classes)):
    print(classes[i], len(files[i]))
    

y = [int(0.75*len(files[i])) for i in range(len(classes))]
z = [len(files[i]) for i in range(len(classes))]
print(z)
print(y)
w = [z[i] - y[i] for i in range(len(classes))]
print(w)

train_files = []
train_label = []
test_label = []
test_files = []
for i in range(len(classes)):
    train_files.extend(files[i][:y[i]])
    test_files.extend(files[i][y[i]: ])
    train_label.extend([i]*y[i])
    test_label.extend([i]*w[i])
    assert(len(train_files) == len(train_label))
    assert(len(test_files) == len(test_label))

X_google_test= [] 
for i in range(len(train_files)):
    image = cv2.imread(train_files[i])
    image = cv2.resize(image, (196, 196), interpolation=cv2.INTER_AREA)
    t1 = np.random.randint(low=-10, high=0)
    t2 = np.random.randint(low=0, high=10)
    translation_matrix1 = np.float32([[1,0,t1], [0,1,0]])
    translation_matrix2 = np.float32([[1,0,t2], [0,1,0]])
    img1 = cv2.warpAffine(image, translation_matrix1, (196, 196))
    img2 = cv2.warpAffine(image, translation_matrix2, (196, 196))
    black1 = cv2.inRange(img1, np.array([0, 0, 0], dtype = np.uint8), np.array([1, 1, 1], dtype = np.uint8))
    black2 = cv2.inRange(img2, np.array([0, 0, 0], dtype = np.uint8), np.array([1, 1, 1], dtype = np.uint8))
    res1 = cv2.bitwise_and(img1, img1, black1)
    res2 = cv2.bitwise_and(img2, img2, black2)
    res1[black1 == 255] = (255, 255, 255)
    res2[black2 == 255] = (255, 255, 255)
    res1 = cv2.flip(res1,1)
    '''
    white_u = np.array([255, 255, 255], dtype=np.uint8)
    white_l = np.array([250, 250, 250], dtype=np.uint8)
    white1 = cv2.inRange(res1, white_l, white_u)
    white2 = cv2.inRange(res2, white_l, white_u)
    noise1 = np.random.randint(255, size=3)
    noise2 = np.random.randint(255, size=3)
    res1[white1 == 255] = noise1
    res2[white2 == 255] = noise2
    '''
    X_google_test.append(res1)
    X_google_test.append(res2)
X_google_test = np.array(X_google_test)/255.
print(sys.getsizeof(X_google_test))
#np.save('X_google_test', X_google_test)
