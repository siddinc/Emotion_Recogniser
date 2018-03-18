'''
from keras import backend as K
import importlib
import os
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend
set_keras_backend("theano")
'''
import keras
import os
import cv2
import numpy
from itertools import repeat
import importlib
from cnn_model import buildCNNModel
#from cnn_model_V2 import buildCnnModel
num_classes = 6


def listALLFiles(path, formats=['png','jpg','jpeg','tif']):
    results = []
    for root,subFolders,files in os.walk(path):
        for file in files:
            if file.split('.')[-1] in formats:
                results.append("/".join([root,file]))
    return results            

def preProcessImage(path,img_width ,img_height):
    img = cv2.imread(path, 0)
    img = cv2.resize(img,(img_width , img_height),interpolation = cv2.INTER_CUBIC)
    img = img.astype('float32')
    img /= 255    
    return img

def prepareData(size):
    input_samples = []
    output_labels = []
    for _class in range(num_classes):
        path = '../dataset/ferDataset/%d' %(_class)
        length = len(os.listdir(path))
        samples = numpy.array(list(map(preProcessImage,listALLFiles(path),repeat(size[0], length),repeat(size[1],length))))
        input_samples.append(samples)
        output_labels.append(numpy.array([_class]*len(samples)))
    inputs = numpy.concatenate(input_samples , axis = 0)  
    outputs = numpy.concatenate(output_labels, axis = 0)
    
    #convert to hot vectors
    output_hot_vectors = numpy.zeros((len(outputs), num_classes))
    output_hot_vectors[numpy.arange(len(outputs)),outputs] = 1
    outputs = output_hot_vectors
    
    #shuffle theinputs and outputs the same way
    p = numpy.random.permutation(len(inputs))
    inputs = inputs[p]
    outputs = outputs[p]
    
    return inputs , outputs

if __name__ == '__main__':
    no_of_epochs = 5
    emotion_models_path = '../trained_model/emotion_models/'
    size = [64,64]
    inputs, outputs = prepareData(size) 
    inputs = inputs.reshape(inputs.shape[0],inputs.shape[1],inputs.shape[2],1)
    num_of_samples = len(inputs)
    train_data_length = int(num_of_samples*0.8)
    x_train, x_test = inputs[0:train_data_length], inputs[train_data_length:]
    y_train, y_test = outputs[0:train_data_length],  outputs[train_data_length:]
    model = buildCNNModel(inputs.shape[1:], num_classes, 32, (3,3), 0.05, (2,2), 1)
    #model = buildCnnModel(inputs.shape[1:], num_classes, 32, (3,3), 0.05, (2,2), 1)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size = 32, epochs = no_of_epochs , validation_data = (x_test,y_test))
    
    if not os.path.exists(emotion_models_path):
        os.makesdirs(emotion_models_path)
    model.save(emotion_models_path + 'emotion_recod_%d_acc-%f.model'%(no_of_epochs,history.history['val_acc'][4]))