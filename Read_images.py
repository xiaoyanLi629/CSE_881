import cv2
import os
from PIL import Image
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


def load_images_from_folder_numpy(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    return images


def rgb_to_gray(img):
    # grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    gray_scale_image = (R + G + B)

    return gray_scale_image


# image_list = load_images_from_folder_numpy('traindata')
# image = image_list[0]
# # plt.imshow(image)
# # plt.show()
#
#
# grayImage = rgb_to_gray(image)
# plt.imshow(grayImage, cmap='gray')
# plt.show()
# image.save('greyscale.png')


def load_images_from_folder(folder):
    images = []
    path = folder + '/*.png'
    i = 1
    for filename in sorted(glob.glob(path)):
        if i % 20 == 0:
            print(filename)
        im = Image.open(filename)
        image = im.convert('LA')
        image = image.resize((300, 300), Image.ANTIALIAS)
        image_array = np.asarray(image)
        image_array_gray = image_array[:, :, 0]
        if image is not None:
            images.append(image_array_gray)
        i = i + 1
    return images


image_list = load_images_from_folder('traindata')
# plt.imshow(image_list[-1], cmap='gray')
# plt.show()

train_data = np.ndarray((len(image_list), image_list[0].shape[0] * image_list[0].shape[1]), int)
for i in range(len(image_list)):
    image_faltten = image_list[i].flatten()
    train_data[i, :] = image_faltten

label_data = pd.read_csv('label.csv', header=None)
label = label_data[:][1]
train_data = np.concatenate((label[:, np.newaxis], train_data), axis=1)
# train_data[:, 0] = train_data[:, 0] + 1


train_data = train_data.astype(int)
np.savetxt('data_set.txt', train_data.astype(int), fmt='%i', delimiter=',')   # X is an array