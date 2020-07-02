import os

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, \
    Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img


class emotion:
    def __init__(self):
        # size of the image: 48*48 pixels
        self.pic_size = 48
        # input path for the images
        self.base_path = "emotion_dataset/"

    def get_data(self):
        # number of images to feed into the NN for every batch
        batch_size = 512

        datagen_train = ImageDataGenerator()
        datagen_validation = ImageDataGenerator()

        self.train_generator = datagen_train.flow_from_directory(self.base_path + "train",
                                                                 target_size=(self.pic_size, self.pic_size),
                                                                 color_mode="grayscale",
                                                                 batch_size=batch_size,
                                                                 class_mode='categorical',
                                                                 shuffle=True)

        self.validation_generator = datagen_validation.flow_from_directory(self.base_path + "val",
                                                                           target_size=(self.pic_size, self.pic_size),
                                                                           color_mode="grayscale",
                                                                           batch_size=batch_size,
                                                                           class_mode='categorical',
                                                                           shuffle=False)

    def model(self):

        # number of possible label values
        nb_classes = len(os.listdir(self.base_path + "train/"))

        # Initialising the CNN
        model = Sequential()

        # 1st Convolution layer
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 2nd Convolution layer
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 3rd Convolution layer
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4th Convolution layer
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Flattening
        model.add(Flatten())

        # Fully connected layer 1st layer
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        # Fully connected layer 2nd layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(nb_classes, activation='softmax'))

        opt = Adam(lr=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        epochs = 10

        checkpoint = ModelCheckpoint("model_weights2.h5")
        callbacks_list = [checkpoint]

        history = model.fit_generator(generator=self.train_generator,
                                      steps_per_epoch=self.train_generator.n // self.train_generator.batch_size,
                                      epochs=epochs,
                                      validation_data=self.validation_generator,
                                      verbose=1,
                                      shuffle=True,
                                      validation_steps=self.validation_generator.n // self.validation_generator.batch_size,
                                      callbacks=callbacks_list
                                      )

        # serialize model structure to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

    def show_data(self):

        plt.figure(0, figsize=(12, 20))
        cpt = 0

        for expression in os.listdir(self.base_path + "train/"):
            for i in range(1, 6):
                cpt = cpt + 1
                plt.subplot(7, 5, cpt)
                img = load_img(
                    self.base_path + "train/" + expression + "/" + os.listdir(self.base_path + "train/" + expression)[
                        i],
                    target_size=(self.pic_size, self.pic_size))
                plt.imshow(img, cmap="gray")

        plt.tight_layout()
        plt.show()


emo = emotion()
emo.show_data()
emo.get_data()
emo.model()
