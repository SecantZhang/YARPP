#!/usr/bin/env python
"""
YARPP - Yet Another Random Points Plot
"""

# Standard libraries
from collections import deque

# Third party libraries
import numpy as np
from numpy.random import multivariate_normal
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class YARPP:
    """
    Class for YARPP
    """

    def __init__(self, input_img_path, threshold=(127, 255), num_objects=1):
        """
        Constructor for YARPP class.
        """
        self.input_img_path = input_img_path
        # TODO: error exceptions for valid input path.
        self.threshold = threshold
        self.num_objects = 1
        self.objects_coord_list = None
        self.max_kmpp_iter = 300
        self.binary_img = self.img_binary_conversion(input_img_path, threshold)
        self.shape = self.binary_img.shape
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

    def k_means_pp(self, img_ary, max_iter=300, early_stopping_iter=10):
        """
        Function for finding the centroids of the image object.
        :param img_ary:                 Binary image array.
        :param max_iter:                Value of max iteration number.
        :param early_stopping_iter      Number of steps for early stopping.
        :return:                        list of centroids of the image object.
        """
        self.objects_coord_list = np.array(
            [[x, y] for x in range(img_ary.shape[0]) for y in range(img_ary.shape[1]) if img_ary[x][y] == 0])
        centroids = self.smart_initialization(self.objects_coord_list, self.num_objects)
        history_queue = deque(maxlen=early_stopping_iter)

        for i in range(max_iter):
            pair_dist = cdist(self.objects_coord_list, centroids)
            Y = np.argmin(pair_dist, axis=1)
            centroids = [tuple(np.mean(self.objects_coord_list[Y == j], axis=0)) for j in range(self.num_objects)]
            if i < early_stopping_iter:
                history_queue.appendleft(centroids)
            else:
                if self.early_stopping(history_queue, 0):
                    print("Clustering converges after {} iterations".format(i))
                    break

        return [(int(value[0]), int(value[1])) for value in centroids]

    def early_stopping(self, es_list, error_range=0):
        if error_range == 0 or error_range == (0, 0):
            for obj_index in range(self.num_objects):
                if len(set([hist_obj[obj_index] for hist_obj in es_list])) != 1:
                    return False
        return True

    def sampling(self, size=300):
        if self.num_objects == 1:
            mean_vec = [self.centroids[0][0], self.centroids[0][1], 120]
            var_vec = [(self.shape[0] / 6) ** 2, (self.shape[1] / 6) ** 2, 60**2]
            cov_mat = np.diag(var_vec)
            # multi_norm_sample = multivariate_normal(mean_vec, cov_mat, size=size)
            # multi_norm_filtered = np.array(
            #     [list(x) for x in multi_norm_sample if 0 <= x[0] < self.shape[0] and 0 <= x[1] < self.shape[1]])
            # return multi_norm_filtered
            multi_norm_sample = []
            success_count = 0
            total_iteration = 0
            while success_count < size:
                curr_sample = multivariate_normal(mean_vec, cov_mat, size=1)[0]
                if 0 < curr_sample[0] < self.shape[0] and 0 < curr_sample[1] < self.shape[1] and curr_sample[2] > 0:
                    multi_norm_sample.append(curr_sample)
                    success_count += 1
                total_iteration += 1

            return np.array(multi_norm_sample)

    def helper_plot(self, data=None, sample_size=300):
        fig, ax = plt.subplots()
        ax.matshow(Z=self.binary_img)
        sampled_scatters = self.sampling(sample_size)
        ax.scatter(x=sampled_scatters[:, 1], y=sampled_scatters[:, 0], c='r', s=sampled_scatters[:, 2])

        for i, txt in enumerate(sampled_scatters[:, 2]):
            ax.annotate(round(txt, 2), (sampled_scatters[i, 1], sampled_scatters[i, 0]))

        plt.show()

    # TODO: Metrapolis-Hasting sampling method.
