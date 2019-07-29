import keras
import pickle
import numpy as np
import pandas as pd
import glob
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import plot_model
from keras import regularizers
from keras.models import load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import regularizers, optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt



def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Load in X and y data
X_train = np.array(load_pickle('X_train_mfcc.pkl')).reshape(-1, 16, 64, 1)
y_train = np.array(load_pickle('y_train_final.pkl'))

X_test = np.array(load_pickle('X_test_mfcc.pkl')).reshape(-1, 16, 64, 1)
y_test = np.array(load_pickle('y_test'))

X_holdout = np.array(load_pickle('X_holdout_mfcc.pkl')).reshape(-1, 16, 64, 1)
y_holdout = np.array(load_pickle('y_holdout'))

# Convert the y labels to categorical values to feed to the CNN
y_test = y_test - 2
y_train = y_train - 2
y_train_hot = to_categorical(y_train, num_classes=4)
y_test_hot = to_categorical(y_test, num_classes=4)



if K.image_data_format() == 'channels_first':
    input_shape = (1 , 64, 26)
else:
    input_shape = (16, 64, 1)


# Model - 65% accuracy
model = Sequential()

model.add(Conv2D(256, (3,3), padding='same',input_shape=(input_shape)))
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=((2,2))))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(MaxPooling2D(pool_size=((2,2))))
model.add(Dropout(0.5))

model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(4))
model.add(Activation('softmax'))
print(model.summary())
opt = keras.optimizers.Adam(lr=0.00001, decay=1e-6)
model.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])
cnnhistory=model.fit(X_train, y_train_hot, batch_size=16, epochs=150, validation_data=(X_test, y_test_hot))

# model.save('final_categorical_model.h5')
# model = load_model('../models/final_categorical_model.h5')




# Plot Training and Test
plt.plot(cnnhistory.history['acc'])
plt.plot(cnnhistory.history['val_acc'])
plt.title('CNN Accuracy MFCC 2D')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('best_cnn_mfcc_2d')
plt.show()
