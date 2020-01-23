import numpy as np
import cv2
import os
import os.path
import pandas as pd
import pickle
from plot_it import Plotter
from itertools import combinations


# The following warping code is copied from my problem set #6 submission
def split_dataset(X, y, p, seed=0):
    shuffler = np.random.permutation(y.size)
    m = shuffler[:int(round(y.size * p))]
    n = shuffler[int(round(y.size * p)):]
    return X[m], y[m], X[n], y[n]
# End of copied warping code


def import_false_images(input_size=(32, 32, 3), c=10, stepsize=16):
    imgs = load_images_from_dir(size=(512, 384))

    img_list = []
    y_windowsize, x_windowsize, _ = input_size
    h, w, d = imgs[0].shape
    for img in imgs:
        for y in range(0, h-y_windowsize+stepsize, stepsize):
            for x in range(0, w-x_windowsize+stepsize, stepsize):
                template = img[y:y+y_windowsize, x:x+x_windowsize]
                img_list.append(template)
                if False:
                    cv2.imshow('template', template.astype(np.uint8))
                    cv2.waitKey(0)

    x_false = np.asarray(img_list, dtype=np.uint8)
    y_false = np.ones((x_false.shape[0]), dtype=np.uint8) * c

    return x_false, y_false


def mean_the_sides(X):
    s, x, y, z = X.shape
    for i in range(s):
        mean = np.mean(X[i], axis=(0, 1))
        mean = mean.astype(np.uint8)
        X[i][:, :8, :] = mean
        X[i][:, -8:, :] = mean
    return X


# Code partially from Yogesh Piazza post @1230
def load_h5(filename='store_df_main.h5'):
    with pd.HDFStore(filename) as store_df_main:
        df_main = store_df_main.get('df_main')

    print('DF shape', df_main.shape)
    return df_main


def store_dataset(dataset, filename='seg_train.pickle'):
    with open(filename, 'wb') as output:
        pickle.dump(dataset, output)


def import_seg_images(pickle_file='seg_train.pickle', reload=False, size=(96, 32), seg_dir='seg_train'):
    if os.path.isfile(pickle_file) and reload is False:
        with open(pickle_file, 'rb') as data:
            imgs = pickle.load(data)
    else:
        print('Getting images for the first time...')
        imgs = load_images_from_dir(data_dir=seg_dir, size=(size[1], size[0]))
        with open(pickle_file, 'wb') as output:
            pickle.dump(imgs, output)

    return np.asarray(imgs)


def add_false_images(X, y, p=.25, c=10):

    img_count = y.size

    shuffler = np.random.permutation(img_count)
    m = shuffler[:int(round(img_count * p))]
    gen_images = []
    x, y, z = X[0].shape

    x2 = int(x / 2)
    y2 = int(y / 2)
    radius = int(3 * x2 / 5)
    kernal = np.ones((3, 3), np.uint8)
    j = -50

    for i, img in enumerate(X[m]):
        img = img * 1.
        mean = np.mean(img, axis=(0, 1))
        n = np.random.normal(mean, 2, (x, y, z))
        mean_img = img.copy()
        mean_img[:, :, :] = n
        gen_images.append(cv2.GaussianBlur(mean_img, (5, 5), 0))

        tmp_img = img.copy()
        noise = cv2.randn(np.zeros(tmp_img.shape), np.zeros(3), np.ones(3)*5)
        tmp_img = (tmp_img + noise).astype(np.uint8)
        gen_images.append(tmp_img)

        tmp_img = img.copy()
        tmp_img = cv2.cvtColor(tmp_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
        gen_images.append(tmp_img)

        tmp_img = img.copy().astype(np.uint8)
        tmp_img = cv2.GaussianBlur(tmp_img, (7, 7), 0)
        gen_images.append(tmp_img)

        tmp_img = img.copy().astype(np.uint8)
        tmp_img = cv2.erode(tmp_img, kernal, iterations=1)
        gen_images.append(tmp_img)

        tmp_img = img.copy().astype(np.uint8)
        tmp_img = cv2.dilate(tmp_img, kernal, iterations=1)
        gen_images.append(tmp_img)

        temp_img = mean_img.copy()
        n = np.random.normal(mean, 5, (x2, y, z))
        temp_img[x2:, :, :] = n + j
        gen_images.append(cv2.GaussianBlur(temp_img, (5, 5), 0))

        temp_img = mean_img.copy()
        n = np.random.normal(mean, 5, (x, y2, z))
        temp_img[:, y2:, :] = n + j
        gen_images.append(cv2.GaussianBlur(temp_img, (5, 5), 0))

        temp_img = mean_img.copy()
        n = np.random.normal(mean, 5, (x2, y2, z))
        temp_img[x2:, y2:, :] = n + j
        gen_images.append(cv2.GaussianBlur(temp_img, (5, 5), 0))

        temp_img = mean_img.copy()
        temp_img[x2:, :y2, :] = n + j
        gen_images.append(cv2.GaussianBlur(temp_img, (5, 5), 0))

        temp_img = mean_img.copy()
        temp_img[:x2, y2:, :] = n + j
        gen_images.append(cv2.GaussianBlur(temp_img, (5, 5), 0))

        temp_img = mean_img.copy()
        temp_img[:x2, :y2, :] = n + j
        gen_images.append(cv2.GaussianBlur(temp_img, (5, 5), 0))

    gen_images = np.asarray(gen_images)

    if False:
        for img in gen_images:
            cv2.imshow('gen', img.astype(np.uint8))
            cv2.waitKey(0)

    y_false = np.ones((gen_images.shape[0]), dtype=np.uint8) * c
    return gen_images.astype(np.uint8), y_false


# The following warping code is copied from my problem set #6 submission
def load_images_from_dir(data_dir='input_images', size=(512, 384)):
    ext=['.png', '.jpeg']
    image_files = [f for f in os.listdir(data_dir) if f.endswith(ext[0]) or f.endswith(ext[1])]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f))) for f in image_files]
    imgs = [cv2.resize(x, size) for x in imgs]

    return imgs
# End of copied warping code


def data_file_plotter(dir='.', filename='my_model', title='Model', axis=(1, 50, .5, 1.)):
    plot = Plotter()
    df = pd.read_csv(dir + '/epoch_data_loss.txt', index_col=0)
    plot.plot(df, filename=dir + '/' + filename + '_loss.png', title=title + ' Loss',
              xaxis='Epochs', yaxis='Loss', axis=axis)
    df = pd.read_csv(dir + '/epoch_data_acc.txt', index_col=0)
    plot.plot(df, filename=dir + '/' + filename + '_acc.png', title=title + ' Accuracy',
              xaxis='Epochs', yaxis='Accuracy', axis=axis)


def seg_area_overlap(boxes):

    y1 = np.min(boxes[:, 0])
    x1 = np.min(boxes[:, 1])
    y2 = np.max(boxes[:, 2])
    x2 = np.max(boxes[:, 3])

    # Return in opencv coords ordering
    return y1, x1, y2, x2


def box_mean_reduce(boxes):
    centroids = []
    for b in boxes:
        centroids.append([b[0] + int(b[5]/2), b[1] + int(b[5]/2)])
    centroids = np.asarray(centroids, dtype=np.float32)
    the_mean = centroids.mean(axis=0)
    the_std = centroids.std(axis=0).mean()
    new_boxes = []
    new_centroids = []

    # Eliminate boxes too far from a set standard deviation from the cluster mean
    for i in range(boxes.shape[0]):
        if np.linalg.norm(the_mean - centroids[i]) < the_std * 4:
            new_boxes.append(boxes[i])
            new_centroids.append(centroids[i])

    boxes = np.asarray(new_boxes)
    centroids = np.asarray(new_centroids)
    # Eliminate boxes that exceed past the average distance between centers of remaining boxes
    dist = [np.linalg.norm(p1 - p2) for p1, p2 in combinations(centroids, 2)]
    if len(dist) > 0:
        average_dist = sum(dist) / len(dist)
    else:
        return None
    new_boxes = []
    r = boxes.shape[0]
    for i in range(r):
        dist = 0
        for j in range(r):
            if i != j:
                dist += np.linalg.norm(centroids[i] - centroids[j])
        dist = dist / (r - 1)
        if dist <= average_dist * 2.5:
            new_boxes.append(boxes[i])
    return np.asarray(new_boxes)


def modified_non_max_suppress(boxes):

    if boxes is None or boxes == []:
        return None

    return_list = []
    blacklisted = []

    for i, box1 in enumerate(boxes):
        if i not in blacklisted:
            for j in range(i+1, boxes.shape[0]):
                if j not in blacklisted:
                    box2 = boxes[j]
                    # Check if they represent the same number
                    if box1[6] == box2[6]:
                        x1 = max(box1[0], box2[0])
                        y1 = max(box1[1], box2[1])
                        x2 = min(box1[2], box2[2])
                        y2 = min(box1[3], box2[3])
                        inner_area = (y2 - y1) * (x2 - x1)

                        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
                        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
                        union_area = box1_area + box2_area - inner_area

                        # If they intersect
                        if inner_area > 0 < union_area:
                            iou = inner_area / float(union_area)

                            # If they intersect over enough of the total area
                            if iou >= .25:
                                if box1[7] > box2[7]:
                                    blacklisted.append(j)
                                else:
                                    blacklisted.append(i)

    for i in range(boxes.shape[0]):
        if i not in blacklisted:
            return_list.append(boxes[i])

    return np.asarray(return_list)


def cv2_import_image(filename, size=None, shrink=.5):

    img = cv2.imread(filename)
    if size is None:
        img = cv2.resize(img, None, fx=shrink, fy=shrink)
    else:
        img = cv2.resize(img, size)
    return img
