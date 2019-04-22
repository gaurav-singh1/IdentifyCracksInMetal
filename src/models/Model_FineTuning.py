from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import os
from keras import callbacks
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.models import Model
from keras import Input

# path to the model weights files.
# weights_path = '/Users/aayush/PycharmProjects/Cracks/vgg16_weights.h5'
top_model_weights_path = '/Users/aayush/PycharmProjects/Cracks/src/models/bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 208, 156


train_data_dir = '/Users/aayush/Downloads/YE358311_Fender_apron/train'
validation_data_dir = '/Users/aayush/Downloads/YE358311_Fender_apron/validation'
nb_train_samples = 2675
nb_validation_samples = 336
epochs = 50
batch_size = 16


# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_height, img_width, 3)))
print('Model loaded.')


print(base_model.summary())

print("model output shape = ",base_model.output_shape)
# build a classifier model to put on top of the convolutional model
top_model = Sequential()

# top_model.add(Flatten(input_shape=base_model.output_shape[1:]))

top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# base_model.add(top_model)



model = Model(input = base_model.input, outputs = top_model(base_model.output))
# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
print(model.summary())
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1. / 255)

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

model_name = 'Crack_Identifier_FineTuning'
model_dir = os.path.join('checkpoints', model_name)
csv_fn = os.path.join(model_dir, 'train_log.csv')
checkpoint_fn = os.path.join(model_dir, 'Crack_Identifier_FineTuning.{epoch:02d}-{val_loss:.2f}.hdf5')
h5_path = '/Users/aayush/PycharmProjects/Cracks/CracksImages.hdf5'
batch_size = 16

checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True)
csv_logger = callbacks.CSVLogger(csv_fn, append=True, separator=';')
tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, batch_size=batch_size,
                                    write_graph=True, write_grads=True, write_images=True)





# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[csv_logger, tensorboard, checkpointer],
    verbose=1
    )

