import os
import cv2
import numpy as np
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
# import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
patch_size = 256
#########################################################################
print("\n##################################################\n")
print("开始加载数据....")
print("\n##################################################\n")
data_dir = '../dataset/module2/result'

patch_size = 256
def data_load(data_dir):
  data1 = []
  data2 = []
  for path, subdirs, files in os.walk(data_dir):
      dirname = path.split(os.path.sep)[-1]
      if dirname == 'images':
        images = os.listdir(path)
        for i,image_name in enumerate(sorted(images)):
          if image_name.endswith('.jpg'):
            image = cv2.imread(os.path.join(path,image_name),1)
            SIZE_X = (image.shape[1] // patch_size) * patch_size
            SIZE_Y = (image.shape[0] // patch_size) * patch_size
            image = Image.fromarray(image)
            image = image.crop((0,0,SIZE_X,SIZE_Y))
            image = np.array(image)
            data1.append(image)
      elif dirname == 'masks':
        images = os.listdir(path)
        for i,image_name in enumerate(sorted(images)):
          if image_name.endswith('.png'):
            image = cv2.imread(os.path.join(path,image_name),1)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            SIZE_X = (image.shape[1] // patch_size) * patch_size
            SIZE_Y = (image.shape[0] // patch_size) * patch_size
            image = Image.fromarray(image)
            image = image.crop((0,0,SIZE_X,SIZE_Y))
            image = np.array(image)
            data2.append(image)
  return data1,data2
image_data,mask_data = data_load(data_dir)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_data[2])
plt.subplot(122)
plt.imshow(mask_data[2])
plt.show()

def add_data_image(data_lists):
  data = []
  for image in data_lists:
    patches_mask = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap

    for i in range(patches_mask.shape[0]):
      for j in range(patches_mask.shape[1]):

        single_patch_mask = patches_mask[i,j,:,:]
        single_patch_mask = scaler.fit_transform(single_patch_mask.reshape(-1,single_patch_mask.shape[-1])).reshape(single_patch_mask.shape)
        #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
        single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.
        data.append(single_patch_mask)
  return data

def add_data_masks(data_lists):
  data = []
  for image in data_lists:
    patches_mask = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap

    for i in range(patches_mask.shape[0]):
      for j in range(patches_mask.shape[1]):

        single_patch_mask = patches_mask[i,j,:,:]

        #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
        single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.
        data.append(single_patch_mask)
  return data


image_dataset = add_data_image(image_data)
masks_dataset = add_data_masks(mask_data)

plt.imshow(image_dataset[0])


import numpy as np
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[2])
plt.subplot(122)
plt.imshow(masks_dataset[2])
plt.show()

image_dataset = np.array(image_dataset)
masks_dataset = np.array(masks_dataset)

print('shap为')
print(masks_dataset.shape)









#########################################################################
# Do the same for all RGB channels in each hex code to convert to RGB
print("\n##################################################\n")
print("开始处理标签数据....")
print("\n##################################################\n")

Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

Road = '#6EC1E4'.lstrip('#')
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

Vegetation =  'FEDD3A'.lstrip('#')
Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

Water = 'E2A929'.lstrip('#')
Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

Unlabeled = '#9B9B9B'.lstrip('#')
Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155

# Now replace RGB to integer values to be used as labels.
# Find pixels with combination of RGB for the above defined arrays...
# if matches then replace all values in that pixel with a specific integer
def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format.
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == Building,axis=-1)] = 0
    label_seg [np.all(label==Land,axis=-1)] = 1
    label_seg [np.all(label==Road,axis=-1)] = 2
    label_seg [np.all(label==Vegetation,axis=-1)] = 3
    label_seg [np.all(label==Water,axis=-1)] = 4
    label_seg [np.all(label==Unlabeled,axis=-1)] = 5

    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels

    return label_seg
labels = []
for i in range(masks_dataset.shape[0]):
  label = rgb_to_2D_label(masks_dataset[i])
  labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)


print("Unique labels in label dataset are: ", np.unique(labels))

#Another Sanity check, view few mages
import numpy as np
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[1])
plt.subplot(122)
plt.imshow(labels[1][:,:,0])
plt.show()

n_classes = len(np.unique(labels))
from keras.utils import to_categorical

# labels_cat = to_categorical(labels, num_classes=4)
labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)

#######################################
# Parameters for model
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss
# from sklearn.utils.class_weight import compute_class_weight

# weights = compute_class_weight('balanced', np.unique(np.ravel(labels,order='C')),
#                               np.ravel(labels,order='C'))
# print(weights)

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

from simple_multi_unet_model import multi_unet_model, jacard_coef

metrics = ['accuracy', jacard_coef]

print("\n##################################################\n")
print("开始构建模型....")
print("\n##################################################\n")
def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()

history1 = model.fit(X_train, y_train,
                     batch_size=16,
                     verbose=1,
                     epochs=100,
                     validation_data=(X_test, y_test),
                     shuffle=False)

# Minmaxscaler
# With weights...[0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]   in Dice loss
# With focal loss only, after 100 epochs val jacard is: 0.62  (Mean IoU: 0.6)
# With dice loss only, after 100 epochs val jacard is: 0.74 (Reached 0.7 in 40 epochs)
# With dice + 5 focal, after 100 epochs val jacard is: 0.711 (Mean IoU: 0.611)
##With dice + 1 focal, after 100 epochs val jacard is: 0.75 (Mean IoU: 0.62)
# Using categorical crossentropy as loss: 0.71

##With calculated weights in Dice loss.
# With dice loss only, after 100 epochs val jacard is: 0.672 (0.52 iou)


##Standardscaler
# Using categorical crossentropy as loss: 0.677
print("\n##################################################\n")
print("开始保存模型....")
print("\n##################################################\n")
model.save('../models/history1.hdf5')
############################################################
# TRY ANOTHE MODEL - WITH PRETRINED WEIGHTS
# Resnet backbone
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
X_train_prepr = preprocess_input(X_train)
X_test_prepr = preprocess_input(X_test)

# define model
model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')

# compile keras model with defined optimozer, loss and metrics
# model_resnet_backbone.compile(optimizer='adam', loss=focal_loss, metrics=metrics)
model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model_resnet_backbone.summary())

history2 = model_resnet_backbone.fit(X_train_prepr,
                                     y_train,
                                     batch_size=16,
                                     epochs=100,
                                     verbose=1,
                                     validation_data=(X_test_prepr, y_test))
model_resnet_backbone.save('../models/history2.hdf5')
# Minmaxscaler
# With weights...[0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]   in Dice loss
# With focal loss only, after 100 epochs val jacard is:
# With dice + 5 focal, after 100 epochs val jacard is: 0.73 (reached 0.71 in 40 epochs. So faster training but not better result. )
##With dice + 1 focal, after 100 epochs val jacard is:
##Using categorical crossentropy as loss: 0.755 (100 epochs)
# With calc. weights supplied to model.fit:

# Standard scaler
# Using categorical crossentropy as loss: 0.74


###########################################################
# plot the training and validation accuracy and loss at each epoch
history = history1
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../images/history1_loss.png')
plt.show()

acc = history.history['jacard_coef']
val_acc = history.history['val_jacard_coef']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.savefig('../images/history1_IoU.png')
plt.show()

history = history2
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../images/history2_loss.png')
plt.show()

acc = history.history['jacard_coef']
val_acc = history.history['val_jacard_coef']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.savefig('../images/history2_IoU.png')
plt.show()

##################################
from keras.models import load_model

model = load_model("../models/history2.hdf5",
                   custom_objects={'dice_loss_plus_2focal_loss': total_loss,
                                   'jacard_coef': jacard_coef})

# IOU
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

# Using built in keras function for IoU
from keras.metrics import MeanIoU

n_classes = 6
# n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#######################################################################
# Predict on a few images

import random

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
# test_img_norm=test_img[:,:,0][:,:,None]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()

#####################################################################