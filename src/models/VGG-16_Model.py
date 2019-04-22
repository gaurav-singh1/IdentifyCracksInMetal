import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import callbacks
import os
from keras.regularizers import l2

# dimensions of our images.
img_width, img_height = 208, 156

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '/Users/aayush/Downloads/YE358311_Fender_apron/train'
validation_data_dir = '/Users/aayush/Downloads/YE358311_Fender_apron/validation'
nb_train_samples =  2675
nb_validation_samples = 336
epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    print("model loaded")

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print("generator initialized")

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    print("bottleneck_features found for training data")

    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    print("training features saved")

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print("generator for validation data initialized")

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)

    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)

    print("validation features saved")


def train_top_model():

    model_name = 'Crack_Identifier'
    model_dir     = os.path.join('checkpoints', model_name)
    csv_fn        = os.path.join(model_dir, 'train_log.csv')
    checkpoint_fn = os.path.join(model_dir, 'Crack_Identifier.{epoch:02d}-{val_loss:.2f}.hdf5')
    h5_path = '/Users/aayush/PycharmProjects/Cracks/CracksImages.hdf5'
    batch_size = 16



    checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True)
    csv_logger = callbacks.CSVLogger(csv_fn, append=True, separator=';')
    tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, batch_size=batch_size,
                                        write_graph=True, write_grads=True, write_images=True)




    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * 1163 + [1] * 1509)
    # train_labels = np.array([0] * 32 + [1] * 32)

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * 168 + [1] * 168)
    # validation_labels = np.array([0] * 32 + [1] * 32)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=[csv_logger, tensorboard, checkpointer],
              verbose=1
              )
    model.save_weights(top_model_weights_path)


# save_bottlebeck_features()

train_top_model()