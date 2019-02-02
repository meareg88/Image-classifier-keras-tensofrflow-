'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, Activation
import keras 

keras.backend.set_image_dim_ordering('tf')
# path to the model weights files.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# dimensions of our images.
img_width, img_height = 150, 150
datapath = os.getcwd()
train_data_dir = datapath + '/data/train'
validation_data_dir = datapath + '/data/validation'
nb_train_samples = 2000
nb_validation_samples = 802
epochs = 3#50
batch_size = 16

#make input_tensor shape tensorflow compatible
input_tensor = Input(shape=(150, 150, 3))

# build the VGG16 network
new_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)
last = new_model.output
#model = applications.VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)
print('Model loaded.')



# build a classifier model to put on top of the convolutional model
top_model = Sequential()

#experiment


top_model.add(Flatten(input_shape=new_model.output_shape[1:]))
top_model.add(Dense(64))
top_model.add(Activation('relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1))
top_model.add(Activation('sigmoid'))
#experiment


#top_model.add(Flatten(input_shape=new_model.output_shape[1:]))
##x = top_model.add(Flatten(input_shape=new_model.output_shape[1:]))(last)
##top_model.add(Flatten(input_shape=model.output_shape[1:]))
#top_model.add(Dense(256, activation='relu'))
#top_model.add(Dropout(0.5))
#top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path, by_name=True)
# add the model on top of the convolutional base

#model.add(top_model)
#model = Model(inputs = top_model.inputs, outputs = top_model(new_model.output))
#experimental
model = Sequential()


for l in new_model.layers:
    model.add(l)

model.add(top_model)

# lock the top conv layers
#for l in model.layers:
#    l.trainable = False

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
#lr=1e-4
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
