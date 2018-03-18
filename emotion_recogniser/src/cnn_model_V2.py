from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

def buildCnnModel(input_shape, num_classes, num_channels,kernel_size, dropout, pool_size, stride):
    model = Sequential()
    model.add(Conv2D(num_channels,kernel_size , padding='valid', input_shape=input_shape, strides=stride))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(num_channels,  kernel_size))
    convout2 = Activation('relu')
    model.add(convout2)
    convout3 = Activation('relu')
    model.add(convout3)
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('relu'))
    model.add(Dropout(dropout/2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model