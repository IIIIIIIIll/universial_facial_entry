import os
import cv2
from PIL import Image
import numpy as np

IMAGE_SIZE = 64
BLACK = [0, 0, 0]
CONVERT_PGM = False  # true if your public face lib contains pgm during first time initialization


def standardize(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    def padding(image):
        h, w = image.shape
        long_edge = max(h, w)
        t, b, l, r = (0, 0, 0, 0)
        if h < long_edge:
            temp = long_edge - h
            t = temp // 2
            b = temp // 2
        elif w < long_edge:
            temp = long_edge - w
            l = temp // 2
            r = temp // 2
        else:
            pass
        return t, b, l, r

    top, bottom, left, right = padding(image)
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


def image_in(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = standardize(image, IMAGE_SIZE, IMAGE_SIZE)
    return image


IMAGE_SIZE = 64
BLACK = [0, 0, 0]


def standardize(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    def padding(image):
        h, w = image.shape
        long_edge = max(h, w)
        t, b, l, r = (0, 0, 0, 0)
        if h < long_edge:
            temp = long_edge - h
            t = temp // 2
            b = temp // 2
        elif w < long_edge:
            temp = long_edge - w
            l = temp // 2
            r = temp // 2
        else:
            pass
        return t, b, l, r

    top, bottom, left, right = padding(image)
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


def image_in(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = standardize(image, IMAGE_SIZE, IMAGE_SIZE)
    return image


images = []
labels = []


def traverse_dir(path):
    print(path)
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))
        if os.path.isdir(abs_path):
            traverse_dir(abs_path)
        else:
            if CONVERT_PGM:
                if file.endswith('.pgm'):
                    filepath, filename = os.path.split(abs_path)
                    out_file = filename[0:-4] + '.jpg'
                    print(filepath,',',filename, ',', out_file)
                    im = Image.open(abs_path)
                    new_path = os.path.join(filepath, out_file)
                    im.save(os.path.join(new_path))
            if file.endswith('.jpg'):
                image = image_in(abs_path)
                images.append(image)
                labels.append(path)
                print("Check " + file.title())
    # print(labels)
    return images, labels


def get_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if label.endswith('target') else 1 for label in labels])
    return images, labels

#im,lb = get_data("/home/yut/ML_CV/keras_learn/Webcam")
# print(lb)
