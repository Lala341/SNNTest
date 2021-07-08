import tensorflow as tf


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

n_norm_samples=1000
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

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(20,20))
for i, (image, label) in enumerate(train_ds.take(5)):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[label.numpy()[0]])
    plt.axis('off')

# Create L2 regularizer
regularizer = regularizers.l2(0.000001)

# Create image data generator
data_gen = ImageDataGenerator(width_shift_range=0.8,height_shift_range=0.8,rotation_range=30,zoom_range=0.1,
shear_range=0.01)
# Get training iterator
iter_train = data_gen.flow(x_train, y_train, batch_size=256)

print(x_test.shape[1:])
initializer="he_uniform"

tf_model = models.Sequential([
    layers.Conv2D(filters=96,kernel_size=(11,11), padding='same', activation='relu', use_bias=False,
    kernel_initializer=initializer, kernel_regularizer=regularizer,input_shape=(32,32,3)),
    layers.AveragePooling2D(2),
    layers.Conv2D(filters=256, kernel_size=(5,5),  activation='relu', padding="same", use_bias=False,
    kernel_initializer=initializer, kernel_regularizer=regularizer),
    layers.AveragePooling2D(2),
    layers.Conv2D(filters=384, kernel_size=(3,3),  activation='relu', padding="same", use_bias=False,
    kernel_initializer=initializer, kernel_regularizer=regularizer),
    layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu', padding="same", use_bias=False,
    kernel_initializer=initializer, kernel_regularizer=regularizer),
    layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding="same", use_bias=False,
    kernel_initializer=initializer, kernel_regularizer=regularizer),
    layers.AveragePooling2D(2),
    layers.Flatten(),
    layers.Dense(4096, activation='relu', use_bias=False, kernel_regularizer=regularizer),
    layers.Dense(4096, activation='relu', use_bias=False, kernel_regularizer=regularizer),
    layers.Dense(10, activation='softmax', use_bias=False, kernel_regularizer=regularizer)
],name="alexnet")

tf_model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
tf_model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_best_only=True,
                                                 monitor='val_accuracy')

#train TensorFlow model
steps_per_epoch = x_train.shape[0] // 256
tf_model.fit(iter_train, steps_per_epoch=steps_per_epoch, epochs=200, callbacks=cp_callback, validation_data=(x_test,y_test))


#Evaluate TensorFlow model
tf_model.evaluate(x_test, y_test)

#Save alexnet_tf_model
models.save_model(tf_model, 'alexnet_tf_model', save_format='h5')

def initializer(shape, dtype=None):
    stddev = np.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)
with CustomObjectScope({'initializer': initializer}):
    tf_model = models.load_model('alexnet_tf_model')

tf_eval_start_time = perf_counter()
tf_model.evaluate(x_test, y_test)
print("TF evaluation:%f" % (perf_counter() - tf_eval_start_time))

few_spike =True
# Create, suitable converter to convert TF model to ML GeNN
converter = (FewSpike(K=10, signed_input=True, norm_data=[x_norm]) if few_spike 
             else RateBased(input_type='poisson', 
                            norm_data=[x_norm],
                            norm_method='data-norm',
                            spike_norm_time=2500))

# Convert and compile ML GeNN model
mlg_model = Model.convert_tf_model(
        tf_model, converter=converter, connectivity_type='procedural',
        dt=1.0, batch_size=1, rng_seed=0, 
        kernel_profiling=True)


time = 10 if few_spike else 2500
mlg_eval_start_time = perf_counter()
acc, spk_i, spk_t = mlg_model.evaluate([x_test], [y_test], time, save_samples=[])
print("MLG evaluation:%f" % (perf_counter() - mlg_eval_start_time))

print('Accuracy of VGG16 GeNN model: {}%'.format(acc[0]))

#ML GeNN model results
neurons = [l.neurons.nrn for l in mlg_model.layers]
raster_plot(spk_i, spk_t, neurons, time=time)

