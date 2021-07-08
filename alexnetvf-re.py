import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    print("tf.config.experimental.get_memory_growth(gpu)")
    print(tf.config.experimental.get_memory_growth(gpu))


from tensorflow.keras import (models, layers, datasets, callbacks, optimizers,
                              initializers, regularizers)
from ml_genn import Model
from tensorflow.keras.utils import CustomObjectScope
from ml_genn.converters import RateBased, FewSpike
from ml_genn.utils import parse_arguments, raster_plot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from six import iteritems
import cv2
from time import perf_counter
from sklearn.preprocessing import OneHotEncoder


#check if tensorflow is running on GPU
print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())

n_norm_samples=2000
#Load Dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

#Resize images
def resize(images):
    images_resized=np.zeros((len(images),64,64,3))
    for i in range(len(images)):
        images_resized[i]=(cv2.resize(images[i], (64,64), interpolation = cv2.INTER_LINEAR))
    return (images_resized)

#x_train= resize(x_train)     
#x_test= resize(x_test)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))


#Divide data
x_train = x_train / 255.0
x_test = x_test / 255.0

# apparently the OneHotEncoder improves the measurements.... it did...
encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()


index_norm=np.random.choice(x_train.shape[0], n_norm_samples, replace=False)
x_norm = x_train[index_norm]
y_norm = y_train[index_norm]

def initializer(shape, dtype=None):
    stddev = np.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)
with CustomObjectScope({'initializer': initializer}):
    tf_model = models.load_model('alexnet_tf_model')

tf_eval_start_time = perf_counter()
tf_model.evaluate(x_test, y_test)
print("TF evaluation:%f" % (perf_counter() - tf_eval_start_time))

few_spike =True
print("RateBased")
# Create, suitable converter to convert TF model to ML GeNN
converter = (RateBased(input_type='poisson', 
                            norm_data=[x_norm],
                            norm_method='spike-norm',
                            spike_norm_time=10, norm_time=10))

# Convert and compile ML GeNN model
mlg_model = Model.convert_tf_model(
        tf_model, converter=converter, connectivity_type='procedural',
        dt=1.0, batch_size=1, rng_seed=0, 
        kernel_profiling=True)


time = 10
mlg_eval_start_time = perf_counter()
acc, spk_i, spk_t = mlg_model.evaluate([x_test], [y_test], time, save_samples=[])
print("MLG evaluation:%f" % (perf_counter() - mlg_eval_start_time))

print('Accuracy of VGG16 GeNN model: {}%'.format(acc[0]))

#ML GeNN model results
neurons = [l.neurons.nrn for l in mlg_model.layers]
raster_plot(spk_i, spk_t, neurons, time=time)

