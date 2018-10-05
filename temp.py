import numpy as np
from tqdm import tqdm
from keras.preprocessing import image

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(270, 270))
    # convert PIL.Image.Image type to 3D tensor with shape (270, 270, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 270, 270, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

train_tensors = paths_to_tensor(['./data-images/5831135989001-343.jpg'])
train_tensors2 = train_tensors.astype('float32')
train_tensors3 = train_tensors2 / 255

print("SHAPE: ")
print(train_tensors3.shape)

print("FIRST PIXEL EXAMPLE: ")
print(train_tensors3[0][0][0])