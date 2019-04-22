from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.regularizers import l2
from keras import regularizers

import os

def build_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(156, 208, 3), kernel_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)), kernel_regularizer=l2(0.0001))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)), kernel_regularizer=l2(0.0001))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    return model




if __name__ == '__main__':

    model_name = 'Crack_Identifier'
    model_dir     = os.path.join('checkpoints', model_name)
    csv_fn        = os.path.join(model_dir, 'train_log.csv')
    checkpoint_fn = os.path.join(model_dir, 'Crack_Identifier.{epoch:02d}-{val_loss:.2f}.hdf5')
    h5_path = '/Users/aayush/PycharmProjects/Cracks/CracksImages.hdf5'
    batch_size = 16

    # X_train, X_test, y_train, y_test = read_data_from_h5file(h5_path)

    model = build_model()


    print(model.summary())


    checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True)
    csv_logger = callbacks.CSVLogger(csv_fn, append=True, separator=';')
    tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, batch_size=10,
                                        write_graph=True, write_grads=True, write_images=True)

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        '/Users/aayush/Downloads/YE358311_Fender_apron/train',  # this is the target directory
        target_size=(156, 208),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    validation_generator = test_datagen.flow_from_directory(
        '/Users/aayush/Downloads/YE358311_Fender_apron/validation',
        target_size=(156, 208),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        epochs=50,
        steps_per_epoch= 2675 // batch_size,
        validation_data=validation_generator,
        validation_steps= 336 // batch_size,
        callbacks= [csv_logger, tensorboard, checkpointer],
        verbose=1)

    # model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1,
    #           callbacks = [csv_logger, tensorboard, checkpointer],
    #           validation_data=(X_test, y_test), shuffle=True)
    #






