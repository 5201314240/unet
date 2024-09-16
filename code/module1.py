import os
import cv2
import shutil
import numpy as np
import pandas as pd

os.environ["SM_FRAMEWORK"] = "tf.keras"
from patchify import patchify
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from tensorflow import keras
# from tensorflow.keras.utils import get_custom_objects

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

from untils1 import *

# 1.1 数据加载
data_dir = '../dataset/module1/jpg'
patch_size = 256

image_data, mask_data = data_load(data_dir, patch_size)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_data[2])
plt.subplot(122)
plt.imshow(mask_data[2])
plt.show()

#######################################################################
# 定位图像中的logo未知，去除logo
remove_logo_data = remove_logo(image_data)

fig, axs = plt.subplots(1, 2, figsize=(14, 12))
axs[0].imshow(image_data[0])
axs[0].axis('off')
axs[1].imshow(remove_logo_data[0])
axs[1].axis('off')
plt.show()

#######################################################################
# 对数据进行均衡化处理
processing_data1 = image_processing1(remove_logo_data)
# 对第一张图像的原图 和 均衡化后的图像进行展示
flg1, axs = plt.subplots(1, 2, figsize=(14, 12))
axs[0].imshow(remove_logo_data[0])
axs[0].axis('off')
axs[0].set_title("Original Image")
axs[1].imshow(processing_data1[0])
axs[1].axis('off')
axs[1].set_title("Histogram Equalization")
plt.show()

# 对第一张图片的直方图 和 均衡化后的图像直方图进行展示
flg2, axs = plt.subplots(1, 2, figsize=(14, 8))
axs[0].set_title("Original Image")
axs[0].hist(remove_logo_data[0].ravel(), 256)
axs[1].set_title("Histogram Equalization")
axs[1].hist(processing_data1[0].ravel(), 256)
plt.show()

#######################################################################
processing_data2 = image_processing2(processing_data1)
# 显示第一张图片的原图 和 双边滤波后的图片
flg3, axs = plt.subplots(1, 2, figsize=(14, 8))
axs[0].imshow(processing_data1[0])
axs[0].axis('off')
axs[0].set_title("Original Image")
axs[1].imshow(processing_data2[0])
axs[1].axis('off')
axs[1].set_title("Bilateral filtering")
plt.show()

#######################################################################

save_paths1 = '../dataset/module1/output/images'
save_paths2 = '../dataset/module1/output/masks'
images_save(processing_data2,'module1',save_paths1)
images_save(mask_data,'module1',save_paths2)