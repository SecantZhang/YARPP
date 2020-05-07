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
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from tqdm import tqdm


class YARPP:
    """
    Class for YARPP
    """

    def __init__(self, input_img_path, threshold=(127, 255), num_objects=1):
        """
        Constructor for YARPP class.
        """
        self.input_img_path = input_img_path
        self.input_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
        self.shape = self.input_img.shape
        # TODO: error exceptions for valid input path.
        self.threshold = threshold
        self.num_objects = 1
        self.objects_coord_list = None
        self.max_kmpp_iter = 300
        self.edge_coord, self.edge_poly = self.__edge_extraction(self.input_img)
        self.binary_img = self.img_binary_conversion(self.input_img, threshold)
        self.centroids = self.k_means_pp(self.binary_img, self.max_kmpp_iter)

    @staticmethod
    def img_binary_conversion(img, threshold):
        """
        Function for converting image to binary image format.
        :param img_path:        Path to the image.
        :param threshold:       Tuple value represents (min_threshold, max_threshold)
        :return:                Converted binary image format.
        """
        _, bi_img = cv2.threshold(img, threshold[0], threshold[1], cv2.THRESH_BINARY)
        # return bi_img
        return np.array([[1 if value == 0 else 0 for value in row] for row in bi_img])

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

    def __edge_extraction(self, orig_img):
        # Apply the laplacian filter to the input binary image.
        laplacian = cv2.Laplacian(orig_img, cv2.CV_64F)
        edge_coord = [[x, y] for y in range(self.shape[0]) for x in range(self.shape[1]) if laplacian[y, x] != 0]
        return edge_coord, Polygon(edge_coord)

    def k_means_pp(self, img_ary, max_iter=300, early_stopping_iter=10):
        """
        Function for finding the centroids of the image object.
        :param img_ary:                 Binary image array.
        :param max_iter:                Value of max iteration number.
        :param early_stopping_iter      Number of steps for early stopping.
        :return:                        list of centroids of the image object.
        """
        self.objects_coord_list = np.array(
            [[x, y] for x in range(self.shape[0]) for y in range(self.shape[1]) if img_ary[x][y] == 0])
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

    def sampling(self, radius_type=None, size=300):
        multi_norm_sample = []
        pbar = tqdm(total=size)
        print("Sampling in progress, user specified number of objects = {}".format(self.num_objects))

        if self.num_objects == 1:
            mean_vec = [self.centroids[0][0], self.centroids[0][1]]
            var_vec = [(self.shape[0] / 6) ** 2, (self.shape[1] / 6) ** 2]
            cov_mat = np.diag(var_vec)
            # multi_norm_sample = multivariate_normal(mean_vec, cov_mat, size=size)
            # multi_norm_filtered = np.array(
            #     [list(x) for x in multi_norm_sample if 0 <= x[0] < self.shape[0] and 0 <= x[1] < self.shape[1]])
            # return multi_norm_filtered

            success_count = 0
            total_iteration = 0
            while success_count < size:
                curr_sample = multivariate_normal(mean_vec, cov_mat, size=1)[0]
                curr_radius = self.radius(centroid=self.centroids[0],
                                          point=(curr_sample[0], curr_sample[1]),
                                          radius_range=(0, 50),
                                          distance_range=(0, euclidean(self.centroids[0], (0, 0))))
                if self.sampling_decision(np.append(curr_sample, curr_radius), multi_norm_sample):
                    curr_sample = np.append(curr_sample, curr_radius)
                    multi_norm_sample.append(curr_sample)
                    success_count += 1
                    pbar.update(1)
                total_iteration += 1
        pbar.close()
        return np.array(multi_norm_sample)

    def sampling_decision(self, point, points_list):
        decision = True
        # Check if the point is inside the canvas.
        if 0 < point[0] < self.shape[0] and 0 < point[1] < self.shape[1] and self.binary_img[int(point[0]), int(point[1])] != 1:
            # 1. euclidean(point[0:2], history_point[0:2]) is the distance between two points.
            # 2. (point[2] + history_point[2]) is the sum of the radius of two circle.
            # 3. if (1) - (2) is less than 0, it means that the two circles intercept.
            # We expect that the sampled points with the relative radius does not intercept with other points.
            if sum([(euclidean(point[0:2], history_point[0:2]) - (point[2] + history_point[2])) < 0 for history_point in points_list]) > 0:
                return False
            if Point(point[1], point[0]).buffer(point[2]).intersects(self.edge_poly):
                return False
        else:
            return False
        return decision

    def helper_plot(self, type, data=None, sample_size=300, annotate=False, random_color=True, fill=True):
        fig, ax = plt.subplots()
        ax.matshow(Z=self.binary_img, cmap=plt.cm.binary, vmin=0, vmax=1)
        if data is None:
            data = self.sampling(size=sample_size)
        if type == "scatter":
            ax.scatter(x=data[:, 1], y=data[:, 0], c='r', s=data[:, 2])
        elif type == "circle":
            if random_color:
                for point in data:
                    curr_circle = plt.Circle(xy=(point[1], point[0]), radius=point[2], color=np.random.rand(3, ), fill=fill)
                    ax.add_artist(curr_circle)
            else:
                for point in data:
                    curr_circle = plt.Circle(xy=(point[1], point[0]), radius=point[2], color='r', fill=False)
                    ax.add_artist(curr_circle)

        if annotate:
            for i, txt in enumerate(data[:, 2]):
                ax.annotate(round(txt, 2), (data[i, 1], data[i, 0]))
        ax.add_artist(plt.Circle(xy=(self.centroids[0][1], self.centroids[0][0]), radius=10, color='y', fill=fill))
        plt.show()

    # TODO: Features: let user provide their own radius function.
    def radius(self, centroid, point, radius_range, distance_range, radius_decay_type="linear", noisy=True):
        # TODO: Improve the equation for calculating distance range.
        radius = None
        curr_distance = euclidean(centroid, point)
        if radius_decay_type == "linear":
            # radius = -1*((radius_range[0]-radius_range[1])/(distance_range[1]-distance_range[0]))*curr_distance+radius_range[1]
            radius = (distance_range[1] - curr_distance) / 20
        else:
            AssertionError("Invalid radius_decay_type value. ")

        return radius

    # TODO: Metrapolis-Hasting sampling method.
