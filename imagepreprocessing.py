import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

K.set_image_dim_ordering('th')
data = np.load('data.npz')
X_train = data['train_X']
X_test = data['test_X']
# normalize inputs from 0-255 to 0-1
rawf = X_train / 255
rawt = X_test / 255
# one hot encode outputs
labels = np_utils.to_categorical(data['train_y'])
num_classes = labels.shape[1]



features = rawf.reshape(rawf.shape[0], 1, 64, 64).astype('float32')
x_test = rawt.reshape(rawt.shape[0], 1, 64, 64).astype('float32')

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(features)

def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1, 64, 64), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = larger_model()
model.fit_generator(datagen.flow(features,labels,batch_size=10),epochs=8,steps_per_epoch=features.shape[0]
                    ,verbose=1)

result = model.predict_classes(x_test)
new_result = []
for i in range(len(result)):
    new_result.append((i+1,result[i]))
lab = ['Id','Label']
df = pd.DataFrame.from_records(new_result, columns=lab)
print(new_result)
df.to_csv('out2.csv',index=False,header=True)