from keras.models import Sequential
from keras.layers import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D

def New_model(n_class):
    model_name = 'CNN_12_DO'

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(128,128,1)))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(32, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(64, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(128, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(128, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())    

    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(512, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(512, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(1028, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    
    model.add(Convolution2D(1028, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(AveragePooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(n_class))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                 )

    return model, model_name