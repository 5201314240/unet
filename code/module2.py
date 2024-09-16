import os
import cv2
import shutil
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans

os.environ["SM_FRAMEWORK"] = "tf.keras"
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from untils1 import *
from untils2 import *

print("\n##################################################\n")
print("程序初始化中....")
print("\n##################################################\n")

images_data_dir = '../dataset/module1/output/images'
masks_data_dir = '../dataset/module1/output/masks'
patch_size = 256

#######################################################################
# 数据加载，并切割
print("\n##################################################\n")
print("开始加载数据....")
print("\n##################################################\n")

image_data = image_load(images_data_dir)
mask_data = image_load(masks_data_dir)
print(f"原始图像数据有{len(image_data)},标记图像数据有{len(mask_data)}")
image_dataset = add_data_image(image_data, patch_size)
masks_dataset = add_data_masks(mask_data, patch_size)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[2])
plt.subplot(122)
plt.imshow(masks_dataset[2])
plt.show()
# 保存切割后的数据集
print("\n##################################################\n")
print("开始保存切割后数据....")
print("\n##################################################\n")
images_save(image_dataset, 'train', '../dataset/module2/splitdata/images')
images_save(masks_dataset, 'masks', '../dataset/module2/splitdata/masks')

print("\n##################################################\n")
print("开始实现数据增强....")
print("\n##################################################\n")
# 设置要保存的batch数量
num_batches_to_save = 2


# 假设扩充合并的图像
def numpy_to_pil(images):
    pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
    return pil_images


combined_pil_images = numpy_to_pil(image_dataset)
combined_pil_masks = numpy_to_pil(masks_dataset)

# Set up data augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True
)

image_save_dir = '../dataset/module2/splitdata/images'
mask_save_dir = '../dataset/module2/splitdata/masks'

os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)

# 数据增强
for i, image in enumerate(combined_pil_images):
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    generator = datagen.flow(image_array, batch_size=1,seed=42)
    for batch_index in range(num_batches_to_save):
        augmented_image = generator.next()[0]
        augmented_pil_image = Image.fromarray((augmented_image * 255).astype(np.uint8))
        if augmented_pil_image.mode != 'RGB':
            augmented_pil_image = augmented_pil_image.convert('RGB')
        save_path = os.path.join(image_save_dir, f'image_{i}_{batch_index}.jpg')
        augmented_pil_image.save(save_path)
print("images 保存成功")

for i, image in enumerate(combined_pil_masks):
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    generator = datagen.flow(image_array, batch_size=1)
    for batch_index in range(num_batches_to_save):
        augmented_image = generator.next()[0]
        augmented_pil_image = Image.fromarray((augmented_image * 255).astype(np.uint8))
        if augmented_pil_image.mode != 'RGB':
            augmented_pil_image = augmented_pil_image.convert('RGB')
        save_path = os.path.join(mask_save_dir, f'mask_{i}_{batch_index}.png')
        augmented_pil_image.save(save_path)
print("masks 保存成功")

#######################################################################
print("\n##################################################\n")
print("开始计算聚类中心点....")
print("\n##################################################\n")
# 计算不同聚类数量的K-means
inertia = []
k_values = range(1, 11)

# 随机选取一张图片 进行评估
num = random.randint(0, len(image_dataset))  # 这个地方可能会出错，越界
pixels = image_dataset[num].reshape(-1, image_dataset[num].shape[-1])  # Flatten image

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixels)
    inertia.append(kmeans.inertia_)

# 绘制肘部法则图 确定聚类中心点 个数
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid()
plt.show()

print("\n##################################################\n")
print("开始构建模型....")
print("\n##################################################\n")
# 由上图可知 n_clusters 最好为3
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

print("\n##################################################\n")
print("开始对结果进行预测....")
print("\n##################################################\n")
# 对聚类后的结果进行保存
kmeans_results = perform_kmeans_clustering_on_images(kmeans, image_dataset[:10])

patch_size = 256
# 获取聚类的第一张图片
images_kmeans = kmeans_results[0]

# 将标签重新整形为原始图像的形状
labels_image = images_kmeans.reshape(patch_size, patch_size)

# 假设 农田 类别 为 1
classes = 1

# 绘制图像
plt.figure(figsize=(12, 8))

# 显示 原始 图片
plt.subplot(2, 2, 1)
plt.imshow(image_dataset[0])
plt.title('Original Image')
plt.axis('off')

# 显示 分组 图片
plt.subplot(2, 2, 2)
plt.imshow(labels_image)
plt.title('Segmented Image')
plt.axis('off')

# 显示 类别 图片
farmland_mask = (labels_image == classes)
plt.subplot(2, 2, 3)
plt.imshow(farmland_mask)
plt.title('Extracted Class 0')
plt.axis('off')

# 在原图上显示类别1
colored_image = np.copy(image_dataset[0])
colored_image[labels_image == classes] = [255, 0, 0]  # 红色
plt.subplot(2, 2, 4)
plt.imshow(colored_image)
plt.title('Extracted Class 1')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\n##################################################\n")
print("程序结束....")
print("\n##################################################\n")
