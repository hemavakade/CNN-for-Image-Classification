from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import json
import h5py
import os
import cPickle
import sys
import pandas as pd
import numpy as np
import glob

def test_mod(model_name, path):

    model = load_model(path + '/' + model_name + '/temp_model.h5')


    img_dir = path + '/' + 'g_test'
    img_height, img_width = 128, 128

    datagen = ImageDataGenerator(
            rescale=1./255,
            fill_mode='constant')

    gen = datagen.flow_from_directory(
            img_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            shuffle=False,  color_mode="grayscale"
            )
    class_dictionary = gen.class_indices 
    class_dict = {v:k for k,v in class_dictionary.items()}    
    actual = gen.classes
    arr = actual[:, np.newaxis]
    predict = model.predict_generator(gen, gen.N)
    final_arr = np.append(arr, arr, axis=1)
    for idx in xrange(len(actual)):
        final_arr[idx,1] = np.argmax(predict[idx])
    df = pd.DataFrame(final_arr, columns=['actual', 'pred']) 
    df.replace({"actual": class_dict, "pred" : class_dict}, inplace=True)
    img_list = map(lambda x: glob.glob(img_dir+'/'+class_dict[x] + '/*'), np.unique(actual))
    img_fl = reduce(lambda x, y: x+y, img_list) 
    df['image_file'] = img_fl
    return df, class_dict
