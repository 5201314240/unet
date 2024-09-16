import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
def image_load(image_paths):
    data = []
    for image in sorted(os.listdir(image_paths)):
        image = cv2.imread(os.path.join(image_paths,image))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)
    return data

def add_data_image(data_lists, patch_size):
    data = []
    for image in data_lists:
        patches_image = patchify(image, (patch_size, patch_size, 3),
                                 step=patch_size)  # Step=256 for 256 patches means no overlap

        for i in range(patches_image.shape[0]):
            for j in range(patches_image.shape[1]):
                single_patch_image = patches_image[i, j, :, :]
                single_patch_image = scaler.fit_transform(
                    single_patch_image.reshape(-1, single_patch_image.shape[-1])).reshape(single_patch_image.shape)
                # single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                single_patch_image = single_patch_image[0]  # Drop the extra unecessary dimension that patchify adds.
                data.append(single_patch_image)
    return data


def add_data_masks(data_lists, patch_size):
    data = []
    for image in data_lists:
        patches_mask = patchify(image, (patch_size, patch_size, 3),
                                step=patch_size)  # Step=256 for 256 patches means no overlap

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]

                # single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                single_patch_mask = single_patch_mask[0]  # Drop the extra unecessary dimension that patchify adds.
                data.append(single_patch_mask)
    return data


def images_save(images, types, save_paths):
    os.makedirs(save_paths, exist_ok=True)
    for i, image in enumerate(images):
        save_path = os.path.join(save_paths, f"{types}_{i}.jpg")
        cv2.imwrite(save_path, image * 255)
    print("保存成功！")

def perform_kmeans_clustering_on_images(kmeans, images):
    kmeans_results = []
    for i, img in enumerate(images):
        pixels = img.reshape(-1, img.shape[-1])  # Flatten image
        kmeans_labels = kmeans.fit_predict(pixels)
        # kmeans_labels = kmeans.predict(pixels)
        kmeans_labels = kmeans_labels.reshape(img.shape[0], img.shape[1])
        kmeans_results.append(kmeans_labels)
        print(f"图片{i}处理成功")
    return kmeans_results