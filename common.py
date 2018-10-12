import numpy as np
import csv
import datetime
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from IPython.display import HTML
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

def initialize():
    random.seed(34283428)
    
def load_image_tensor(img_path, images_size=270):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(images_size, images_size))
    # For grayscale images use:
    # img = image.load_img(img_path, grayscale=False, color_mode='grayscale', target_size=(images_size, images_size))
    # convert PIL.Image.Image type to 3D tensor with shape (images_size, images_size, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, images_size, images_size, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def load_image_tensors(img_paths, images_size=270):
    list_of_tensors = [load_image_tensor(img_path, images_size) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def display_images(image_paths, images_size=270):
    html_content = "<div width='100%'>"
    for image_path in image_paths:
        html_content += '<div style="font-size: 10px; display:inline-block; width: {images_size}px; border:1px solid black"> \
         {image_path}: \
         <img src="{image_path}" style="display:inline-block;"> </div>'.format(images_size=images_size, image_path=image_path)
    html_content += '</div>'
    display(HTML(html_content))    
    
def load_and_split_data(images_size=270):
    print(".. loading & splitting data ..")
    section_start_time = datetime.datetime.utcnow()

    image_to_label = {}
    images_list = []
    label_list = []
    with open('./data-labels/images.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_name = row['IMAGE FILENAME'].strip()
            is_positive = 1 if row['IS POSITIVE'] else 0
            if image_name:
                image_path = './data-images/{image_name}'.format(image_name=image_name)
                image_to_label[image_path] = is_positive
                images_list.append(image_path)
                label_list.append(is_positive)

    label_list_categorical = to_categorical(label_list)

    X_train, X_test, y_train, y_test = train_test_split(np.array(images_list),
                                                        np.array(label_list_categorical),
                                                        test_size=0.20,
                                                        random_state=42)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train,
                                                                y_train,
                                                                test_size=0.20,
                                                                random_state=42)

    train_tensors = load_image_tensors(X_train, images_size).astype('float32') / 255
    test_tensors = load_image_tensors(X_test, images_size).astype('float32') / 255
    valid_tensors = load_image_tensors(X_validate, images_size).astype('float32') / 255

    duration_loading = (datetime.datetime.utcnow() - section_start_time).total_seconds()    
    
    return (train_tensors, X_train, y_train,
            test_tensors, X_test, y_test, 
            valid_tensors, X_validate, y_validate, 
            duration_loading)
    
    
def train_single_model(model, 
                       best_model_filepath, 
                       train_tensors, y_train, 
                       valid_tensors, y_validate, 
                       training_epochs, 
                       batch_size):

    print(".. training the model ..")
    section_start_time = datetime.datetime.utcnow()

    checkpointer = ModelCheckpoint(filepath=best_model_filepath,
                                   verbose=1,
                                   save_best_only=True)

    model.fit(train_tensors, y_train,
              validation_data=(valid_tensors, y_validate),
              epochs=training_epochs,
              batch_size=batch_size,
              callbacks=[checkpointer],
              verbose=1)

    duration_training = (datetime.datetime.utcnow() - section_start_time).total_seconds()        
    return duration_training
    
    
def summarize_model_performance(X_data, y_data, predictions):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    all_positives = 0
    all_negatives = 0
    all_data = len(X_data)

    false_positive_images = []
    false_negative_images = []
    
    for n, data_image in enumerate(X_data):
        prediction_label = True if predictions[n] else False
        truth_label = True if y_data[n][1] else False

        if truth_label:
            all_positives += 1
        else:
            all_negatives += 1

        if prediction_label:
            if truth_label:
                true_positives += 1
            else:
                false_positives += 1
                false_positive_images.append(data_image)
        else:
            if truth_label:
                false_negatives += 1
                false_negative_images.append(data_image)
            else:
                true_negatives += 1

    print("all: ", all_data)
    print("all_positives: ", all_positives)
    print("all_negatives: ", all_negatives)
    print("true_positives: ", true_positives)
    print("true_negatives: ", true_negatives)
    print("false_positives: ", false_positives)
    print("false_negatives: ", false_negatives)

    recall = true_positives / all_positives
    specificity = true_negatives / all_negatives
    accuracy = (true_positives + true_negatives) / all_data
    precision = true_positives / (true_positives + false_positives)
    fp_rate = false_positives / (false_positives + true_negatives)
    fn_rate = false_negatives / (false_negatives + true_positives)
    f1 = 2 * precision * recall / (precision + recall)

    print("RECALL: {0:.2f}".format(recall))
    print("SPECIFICITY: {0:.2f}".format(specificity))
    print("ACCURACY: {0:.2f}".format(accuracy))
    print("PRECISION: {0:.2f}".format(precision))
    print("F1 SCORE: {0:.2f}".format(f1))
    print("FP RATE / ERROR I: {0:.2f}".format(fp_rate))
    print("FN RATE / ERROR II: {0:.2f}".format(fn_rate))
    
    return (false_positive_images, false_negative_images)
                