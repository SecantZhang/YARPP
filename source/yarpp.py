#!/usr/bin/env python
"""
YARPP - Yet Another Random Points Plot
"""

# Standard libraries


# Third party libraries
import numpy as np
import cv2
from scipy.spatial.distance import cdist


class YARPP:
    """
    Class for YARPP
    """
    def __init__(self, input_img_path, threshold=(127, 255), num_objects=1):
        """
        Constructor for YARPP class.
        """
        self.input_img_path = input_img_path
        self.threshold = threshold
        self.num_objects = 1
        self.objects_coord_list = None
        self.max_kmpp_iter = 300
        self.binary_img = self.img_binary_conversion(input_img_path, threshold)
        self.centroids = self.k_means_pp(self.binary_img, self.max_kmpp_iter)

    @staticmethod
    def img_binary_conversion(img_path, threshold):
        """
        Function for converting image to binary image format.
        :param img_path:        Path to the image.
        :param threshold:       Tuple value represents (min_threshold, max_threshold)
        :return:                Converted binary image format.
        """
        img = cv2.imread(img_path, 2)
        _, bi_img = cv2.threshold(img, threshold[0], threshold[1], cv2.THRESH_BINARY)
        return bi_img

    @staticmethod
    def smart_initialization(data, K):
        """
        Function for original smart initialization for the K-means++ algorithm.
        :param data:        Original data.
        :param K:           Number of clusters.
        :return:            List of centroids selected by the smart_initialization.
        """
        data = [list(x) for x in list(data)]
        centroids = [data[int(np.random.choice(len(data), 1))]]
        for _ in range(1, K):
            D2 = np.array(
                [min([np.inner(np.array(c) - np.array(x), np.array(c) - np.array(x)) for c in centroids]) for x in
                 data])
            probs = D2 / D2.sum()
            cumprobs = probs.cumsum()
            r = np.random.rand()
            i = -1
            for j, p in enumerate(cumprobs):
                if r < p:
                    i = j
                    break
            centroids.append(data[i])
        return np.array(centroids)

    def k_means_pp(self, img_ary, max_iter=300):
        """
        Function for finding the centroids of the image object.
        :param img_ary:         Binary image array.
        :param max_iter:        Value of max iteration number.
        :return:                list of centroids of the image object.
        """
        self.objects_coord_list = np.array([[x, y] for x in range(img_ary.shape[0]) for y in range(img_ary.shape[1]) if img_ary[x][y] == 0])
        centroids = self.smart_initialization(self.objects_coord_list, self.num_objects)
        for i in range(max_iter):
            pair_dist = cdist(self.objects_coord_list, centroids)
            Y = np.argmin(pair_dist, axis=1)
            for j in range(self.num_objects):
                centroids[j] = np.mean(self.objects_coord_list[Y == j], axis=0)

        return centroids
