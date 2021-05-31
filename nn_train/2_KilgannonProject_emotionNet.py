"""
 
CISC 642
Final Project
CNN for emotional detection
Jon Kilgannon
 
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
 

batch_size = 32
epochs_wanted = 200
base_size = 16

data_location = "D:\\!data\\college\\2021Spring\\fer2103\\"

# Define the network. Make it small so it runs fast.
model = Sequential()
model.add(Conv2D(base_size, (3, 3), input_shape=(48, 48, 1)))
model.add(Activation('relu'))

model.add(Conv2D(base_size*2, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.1))

model.add(Conv2D(base_size*4, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(base_size*4))
model.add(Activation('relu'))

model.add(Dropout(0.3))

model.add(Dense(7))             # We have seven emotions
model.add(Activation('softmax'))


# sparse_categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr = 1e-3),
              metrics=['accuracy'])

print(model.summary())

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        data_location + "train",
        target_size=(48,48),
        color_mode = "grayscale",
        batch_size=batch_size,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        data_location + "test",
        target_size=(48,48),
        color_mode = "grayscale",
        batch_size=batch_size,
        class_mode='categorical')

monitor_type = 'loss'

# If we're not improving, stop.
EARLYSTOP = EarlyStopping(patience=30, 
                          monitor=monitor_type, 
                          restore_best_weights=True)

# Save off the very best model we can find; avoids overfitting.
best_model_loc = './best_model_incremental_{epoch}.h5'
CHKPT = ModelCheckpoint(best_model_loc, 
                     monitor=monitor_type, 
                     mode='min', 
                     verbose=1, 
                     save_best_only=True)

history = model.fit_generator(train_generator,
                    steps_per_epoch=28709 // batch_size,
                    shuffle=True,
                    epochs=epochs_wanted,
                    validation_data=validation_generator,
                    validation_steps=7178 // batch_size,
                    callbacks=[EARLYSTOP, CHKPT])

model.save('emotions.h5')

print("Done")

