import tensorflow
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



ImageWidth = 100
ImageHeight = 100
ImageSize = (ImageWidth, ImageHeight)
ImageChannels = 3
BatchSize = 16
Epochs = 500


def createLabel():
    FileName = os.listdir("Train")
    Categories = []

    for filename in FileName:
        category = filename.split('_')[0]
        if category == 'Basketball':
            Categories.append(0)
        elif category == 'Football':
            Categories.append(1)
        elif category == 'Rowing':
            Categories.append(2)
        elif category == 'Swimming':
            Categories.append(3)
        elif category == 'Tennis':
            Categories.append(4)
        else:
            Categories.append(5)

    df = pd.DataFrame({
        'ImageName': FileName,
        'Label': Categories
    })
    df["Label"] = df["Label"].replace(
        {0: 'Basketball', 1: 'Football', 2: 'Rowing', 3: 'Swimming', 4: 'Tennis', 5: 'Yoga'})

    return df


def Training(train_df):
    train_df = train_df.reset_index(drop=True)

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        "Train",
        x_col='ImageName',
        y_col='Label',
        target_size=ImageSize,
        class_mode='categorical',
        batch_size=BatchSize,
    )

    return train_generator


def Validation(validate_df):
    validate_df = validate_df.reset_index(drop=True)

    validation_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1, )

    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        "Train",
        x_col='ImageName',
        y_col='Label',
        target_size=ImageSize,
        class_mode='categorical',
        batch_size=BatchSize,
    )

    return validation_generator


def Testing():
    TestFileName = os.listdir("NTest")
    TestData = pd.DataFrame({'ImageName': TestFileName})
    Samples = TestData.shape[0]

    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_dataframe(
        TestData,
        "NTest",
        x_col='ImageName',
        y_col=None,
        class_mode=None,
        target_size=ImageSize,
        batch_size=BatchSize,
        shuffle=False,
    )
    predict = model.predict(test_generator, steps=np.ceil(Samples / BatchSize))

    TestData['Label'] = np.argmax(predict, axis=-1)

    label_map = dict((v, k) for k, v in train_generator.class_indices.items())
    TestData['Label'] = TestData['Label'].replace(label_map)
    TestData['Label'] = TestData['Label'].replace(
        {'Basketball': 0, 'Football': 1, 'Rowing': 2, 'Swimming': 3, 'Tennis': 4, 'Yoga': 5})

    SubmissionFile = TestData.copy()
    SubmissionFile['image_name'] = SubmissionFile['ImageName']
    SubmissionFile['label'] = SubmissionFile['Label']
    SubmissionFile.drop(['ImageName', 'Label'], axis=1, inplace=True)
    SubmissionFile.to_csv('submission.csv', index=False)


def CNN_Model():
    model = Sequential()

    model.add(
        Conv2D(32, (5, 5), activation='relu', input_shape=(ImageWidth, ImageHeight, ImageChannels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


df = createLabel()

TrainData, ValidateData = train_test_split(df, test_size=0.20, random_state=1)

train_generator = Training(TrainData)
validation_generator = Validation(ValidateData)

TotalValidate = ValidateData.shape[0]
TotalTrain = TrainData.shape[0]

model = CNN_Model()

if os.path.exists('model.h5'):
    model.load_weights('model.h5')
else:
    history = model.fit(
        train_generator,
        epochs=Epochs,
        validation_data=validation_generator,
        validation_steps=TotalValidate // BatchSize,
        steps_per_epoch=TotalTrain // BatchSize,
    )
    model.save_weights('model.h5')

Testing()

