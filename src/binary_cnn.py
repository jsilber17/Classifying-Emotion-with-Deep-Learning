import keras
import pickle
import numpy as np
import pandas as pd
import glob
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers import MaxPooling2D, Sequential, Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras import optimizers
from keras.models import Model
from keras.models import load_model


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizers.rmsprop(lr=0.00005, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])
# model.save('final_binary_model.h5')


#EC2 INSTANCE
def create_dataframes_for_generator(directory_name, label_position, img_name_position):
    ''' Create dataframes that contain the image name and label of the spectograms '''

    directory_name = np.array(glob(glob(directory_name)))
    namelst = []
    label = []

    for name in directory_name:
        if name[label_position] == '2' or name[label_position] == '3':
            namelst.append(name[img_name_position:])
            label.append('1')
        elif name[label_position] == '4' or name[label_position] == '5':
            namelst.append(name[img_name_position:])
            label.append('0')
        else:
            pass
    df = pd.DataFrame([namelst, label]).T
    df.columns = ['name', 'label']
    return df

df_train = create_dataframes_for_generator('img_train/*', 26, 10)
df_test = create_dataframes_for_generator('img_test/*', 24, 9)
df_final = pd.concat([df_train, df_test], ignore_index=True)


train_datagen=ImageDataGenerator(rescale=1.0/255.0)
test_datagen=ImageDataGenerator(rescale=1.0/255.0)


# Define generators for test and train data
train_generator=train_datagen.flow_from_dataframe(
    dataframe=df_final[:1400],
    directory="img_train",
    x_col="name",
    y_col="label",
    batch_size=1000,
    shuffle=True,
    class_mode="binary",
    target_size=(64,64))

test_generator=test_datagen.flow_from_dataframe(
    dataframe=df_final[1400:],
    directory="img_test",
    x_col="name",
    y_col="label",
    batch_size=1000,
    shuffle=True,
    class_mode='binary',
    target_size=(64,64))

history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=500,
    validation_data=test_generator,
    validation_steps=10)

# Plot Training and Test
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('CNN Binary Accuracy Spectogram')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_performances_binary')
plt.show()
model.save('final_binary_model_pos_neg.h5')
