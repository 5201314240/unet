import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()


def data_load(data_dir, patch_size):
    data1 = []
    data2 = []
    for path, subdirs, files in os.walk(data_dir):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':
            images = os.listdir(path)
            for i, image_name in enumerate(sorted(images)):
                if image_name.endswith('.jpg'):
                    image = cv2.imread(os.path.join(path, image_name), 1)
                    SIZE_X = (image.shape[1] // patch_size) * patch_size
                    SIZE_Y = (image.shape[0] // patch_size) * patch_size
                    image = Image.fromarray(image)
                    image = image.crop((0, 0, SIZE_X, SIZE_Y))
                    image = np.array(image)
                    data1.append(image)
        elif dirname == 'masks':
            images = os.listdir(path)
            for i, image_name in enumerate(sorted(images)):
                if image_name.endswith('.png'):
                    image = cv2.imread(os.path.join(path, image_name), 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    SIZE_X = (image.shape[1] // patch_size) * patch_size
                    SIZE_Y = (image.shape[0] // patch_size) * patch_size
                    image = Image.fromarray(image)
                    image = image.crop((0, 0, SIZE_X, SIZE_Y))
                    image = np.array(image)
                    data2.append(image)
    return data1, data2


def remove_logo(images):
    data = []
    for image in images:
        # 获取图像的高度和宽度
        height = image.shape[0]
        width = image.shape[1]

        x1 = width - 100  # 矩形左上角的x坐标
        y1 = height - 50  # 矩形左上角的y坐标
        x2 = width  # 矩形右下角的x坐标
        y2 = height  # 矩形右下角的y坐标
        logo_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(logo_mask, (x1, y1), (x2, y2), 255, -1)

        # 使用inpaint函数修复图像
        inpainted_image = cv2.inpaint(image, logo_mask, 3, cv2.INPAINT_TELEA)
        data.append(inpainted_image)
    return data


# 对数据进行均衡化处理
def image_processing1(images):
    # 存储出后的的图像数据
    data = []
    # 遍历图像
    for image in images:
        # 分离图像到不同的颜色通道
        channels = cv2.split(image)
        # 对每个通道进行直方图均衡化
        eq_channels = []
        for ch in channels:
            eq_ch = cv2.equalizeHist(ch)
            eq_channels.append(eq_ch)
        # 合并均衡化的通道
        eq_img = cv2.merge(eq_channels)
        data.append(eq_img)
    return data


# 对图像进行双边滤波
def image_processing2(images):
    data = []
    # 遍历图像
    for image in images:
        bilateral = cv2.bilateralFilter(image, 55, 100, 100)
        data.append(bilateral)
    return data


# 定义图片保存函数
def images_save(images, types, save_paths):
    os.makedirs(save_paths, exist_ok=True)
    for i, image in enumerate(images):
        save_path = os.path.join(save_paths, f"{types}_{i}.jpg")
        cv2.imwrite(save_path, image * 255)
    print("保存成功！")
