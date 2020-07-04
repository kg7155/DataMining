# ID=83b
import os
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

#image dimensions
img_height, img_width = 64, 64
image_dim = (img_height, img_width, 1)

#Model parameters
epochs = 1
num_classes = 4
batch_size = 128

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=image_dim))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Data augmentation
train_data_aug = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)

real_data_aug = ImageDataGenerator(
    rescale=1./255)

#Data loaders
train_gen = train_data_aug.flow_from_directory(
        "data/train",
        target_size=(img_height, img_width),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

validation_gen = real_data_aug.flow_from_directory(
        "data/validation",
        target_size=(img_height, img_width),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical")

test_gen = real_data_aug.flow_from_directory(
        "data/test",
        target_size=(img_height, img_width),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        )

#Train
model.fit_generator(
    train_gen,
    epochs=epochs,
    validation_data=validation_gen,
    verbose=1
)

#Predict
predictions = model.predict_generator(test_gen, verbose=1)
predictions = np.argmax(predictions, axis=1)

#Save to output file
pd.DataFrame(predictions).to_csv("output.csv", header=False, index=False)