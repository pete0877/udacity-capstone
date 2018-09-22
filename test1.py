import csv
import datetime
import os
import random

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm

random.seed(34283428)
base_dir_path = os.path.dirname(os.path.realpath(__file__))
best_model_filepath = '{}/saved-models/model1.hdf5'.format(base_dir_path)
training_epochs = 1

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

start_time = datetime.datetime.utcnow()
print(".. loading & splitting data ..")

image_to_label = {}
images_list = []
label_list = []
with open('{base_dir_path}/data-labels/images.csv'.format(base_dir_path=base_dir_path),
          newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['IMAGE FILENAME'].strip()
        is_huddle = 1 if row['IS HUDDLE'] else 0
        if image_name:
            image_path = '{base_dir_path}/data-images/{image_name}'.format(base_dir_path=base_dir_path,
                                                                           image_name=image_name)
            image_to_label[image_path] = is_huddle
            images_list.append(image_path)
            label_list.append(is_huddle)

label_list_categorical = to_categorical(label_list)

X_train, X_test, y_train, y_test = train_test_split(np.array(images_list),
                                                    np.array(label_list_categorical),
                                                    test_size=0.20,
                                                    random_state=42)

X_train, X_validate, y_train, y_validate = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.20,
                                                            random_state=42)

train_tensors = paths_to_tensor(X_train).astype('float32')
test_tensors = paths_to_tensor(X_test).astype('float32')
valid_tensors = paths_to_tensor(X_validate).astype('float32')

loading_finished_time = datetime.datetime.utcnow()
duration_loading = (loading_finished_time - start_time).total_seconds()
print(".. constructing the model ..")

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding='same', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(".. training the model ..")

checkpointer = ModelCheckpoint(filepath=best_model_filepath,
                               verbose=1,
                               save_best_only=True)

model.fit(train_tensors, y_train,
          validation_data=(valid_tensors, y_validate),
          epochs=training_epochs,
          batch_size=20,
          callbacks=[checkpointer],
          verbose=1)

training_finished_time = datetime.datetime.utcnow()
duration_training = (training_finished_time - loading_finished_time).total_seconds()

print(".. loading best weights ..")

model.load_weights(best_model_filepath)

print(".. testing the model ..")

tmp_predictions = [model.predict(np.expand_dims(tensor, axis=0)) for tensor in test_tensors]
test_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

testing_finished_time = datetime.datetime.utcnow()
duration_testing = (testing_finished_time - training_finished_time).total_seconds()

print(".. reporting results ..")

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
all_positives = 0
all = len(X_test)

for n, test_image in enumerate(X_test):
    prediction_label = True if test_predictions[n] else False
    truth_label = True if y_test[n][1] else False
    correct_prediction = prediction_label == truth_label

    print("{indicator} {truth_icon} {test_image} (is {truth} predicted {prediction})".format(
        indicator=' ' if correct_prediction else 'X',
        truth_icon='|' if truth_label else '=',
        test_image=test_image,
        truth=truth_label,
        prediction=prediction_label,
    ))

    if truth_label:
        all_positives += 1

    if prediction_label:
        if truth_label:
            true_positives += 1
        else:
            false_positives += 1
    else:
        if truth_label:
            false_negatives += 1
        else:
            true_negatives += 1

print("loading duration: {0:.1f} seconds".format(duration_loading))
print("training duration: {0:.1f} seconds".format(duration_training))
print("testing duration: {0:.1f} seconds".format(duration_testing))

print("all: ", all)
print("all_positives: ", all_positives)
print("true_positives: ", true_positives)
print("true_negatives: ", true_negatives)
print("false_positives: ", false_positives)
print("false_negatives: ", false_negatives)

print("RECALL: {0:.2f}%".format(100 * true_positives / all_positives))
print("ACCURACY: {0:.2f}%".format(100 * (true_positives + true_negatives) / all))
