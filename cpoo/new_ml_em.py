#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image
import ImageColor
import random

import math
import numpy as np
#from numpy import *

from datetime import datetime


BLUE = ImageColor.getrgb(u'Blue')
BLACK = ImageColor.getrgb(u'Black')
RED = ImageColor.getrgb(u'Red')


def ml_em(file_name, K=5):
    image = Image.open(file_name)
    k = 2
    now = datetime.now()
    X = init_features(image)
    u, alpha, sigma = init_distribution_params(X, k, False)
    print 'u:', u
    print 'alpha:', alpha
    print 'sigma:', sigma
    after = datetime.now()
    print 'diff:', after - now
    return image


# image -> X
def init_features(image, d=4):
    width, height = image.size
    X = np.empty([height, width, d])
    for ind_h in range(height):
        for ind_w in range(width):
            pixel = image.getpixel((ind_w, ind_h))
            r_g = pixel[0] - pixel[1]
            r_b = pixel[0] - pixel[2]
            X[ind_h][ind_w] = [ind_w, ind_h, r_g, r_b]

    return X


# X, k -> u, alpha, sigma
def init_distribution_params(X, k, with_noise):
    u, alpha = init_means_weights(X, k, with_noise)
    d = X.shape[2]
    sigma = init_covariance_matrices(d, k)
    return u, alpha, sigma


# X, k -> u, alpha
def init_means_weights(X, k, with_noise):
    height, width, d = X.shape
    X_count = width * height
    u = np.empty([k, d])
    alpha = np.empty(k)
    if k == 2:
        X1_subarray = X[height/4:3*height/4, width/4:3*width/4]
        X1_count = X1_subarray.shape[0] * X1_subarray.shape[1]
        X2_count = X_count - X1_count
        u1 = np.sum(np.sum(X1_subarray, 1), 0)
        u2 = np.sum(np.sum(X, 1), 0) - u1
        u1 /= X1_count
        u2 /= X2_count
        u[:] = [u1, u2]
        alpha[:] = [X1_count, X2_count]
        alpha /= X_count
    elif k == 3:
        X1_subarray_left = X[height/4:3*height/4, width/4:width/2]
        X1_subarray_right = X[height/4:3*height/4, width/2:3*width/4]
        X1_subarray_left_count = X1_subarray_left.shape[0] * X1_subarray_left.shape[1]
        X1_subarray_right_count = X1_subarray_right.shape[0] * X1_subarray_right.shape[1]

        X1_count = X1_subarray_left_count + X1_subarray_right_count
        u1_left = np.sum(np.sum(X1_subarray_left, 1), 0)
        u1_right = np.sum(np.sum(X1_subarray_right, 1), 0)
        u1 = u1_left + u1_right
        X2_count = X_count / 2 - X1_subarray_left_count
        u2 = np.sum(np.sum(X[:,:width/2], 1), 0) - u1_left
        X3_count = X_count / 2 - X1_subarray_right_count
        u3 = np.sum(np.sum(X[:,width/2:], 1), 0) - u1_right
        u1 /= X1_count
        u2 /= X2_count
        u3 /= X3_count
        u[:] = [u1, u2, u3]
        alpha[:] = [X1_count, X2_count, X3_count]
        alpha /= X_count
    elif k == 4:
        X1 = X[:height/2, :width/2]
        X2 = X[:height/2, width/2:]
        X3 = X[height/2:, :width/2]
        X4 = X[height/2:, width/2:]

        X1_count = X1.shape[0] * X1.shape[1]
        u1 = np.sum(np.sum(X1, 1), 0)
        X2_count = X2.shape[0] * X2.shape[1]
        u2 = np.sum(np.sum(X2, 1), 0)
        X3_count = X3.shape[0] * X3.shape[1]
        u3 = np.sum(np.sum(X3, 1), 0)
        X4_count = X4.shape[0] * X4.shape[1]
        u4 = np.sum(np.sum(X4, 1), 0)
        u1 /= X1_count
        u2 /= X2_count
        u3 /= X3_count
        u4 /= X4_count
        u[:] = [u1, u2, u3, u4]
        alpha[:] = [X1_count, X2_count, X3_count, X4_count]
        alpha /= X_count
    elif k == 5:
        X1_subarray_top_left = X[height/4:height/2, width/4:width/2]
        X1_subarray_top_right = X[height/4:height/2, width/2:3*width/4]
        X1_subarray_bottom_left = X[height/2:3*height/4, width/4:width/2]
        X1_subarray_bottom_right = X[height/2:3*height/4, width/2:3*width/4]
        X1_subarray_top_left_count = X1_subarray_top_left.shape[0] * X1_subarray_top_left.shape[1]
        X1_subarray_top_right_count = X1_subarray_top_right.shape[0] * X1_subarray_top_right.shape[1]
        X1_subarray_bottom_left_count = X1_subarray_bottom_left.shape[0] * X1_subarray_bottom_left.shape[1]
        X1_subarray_bottom_right_count = X1_subarray_bottom_right.shape[0] * X1_subarray_bottom_right.shape[1]

        X1_count = (X1_subarray_top_left_count + X1_subarray_top_right_count +
                    X1_subarray_bottom_left_count + X1_subarray_bottom_right_count)
        u1_top_left = np.sum(np.sum(X1_subarray_top_left, 1), 0)
        u1_top_right = np.sum(np.sum(X1_subarray_top_right, 1), 0)
        u1_bottom_left = np.sum(np.sum(X1_subarray_bottom_left, 1), 0)
        u1_bottom_right = np.sum(np.sum(X1_subarray_bottom_right, 1), 0)
        u1 = u1_top_left + u1_top_right + u1_bottom_left + u1_bottom_right
        X2_count = X_count / 4 - X1_subarray_top_left_count
        u2 = np.sum(np.sum(X[:height/2,:width/2], 1), 0) - u1_top_left
        X3_count = X_count / 4 - X1_subarray_top_right_count
        u3 = np.sum(np.sum(X[:height/2,width/2:], 1), 0) - u1_top_right
        X4_count = X_count / 4 - X1_subarray_bottom_left_count
        u4 = np.sum(np.sum(X[height/2:,:width/2], 1), 0) - u1_bottom_left
        X5_count = X_count / 4 - X1_subarray_bottom_right_count
        u5 = np.sum(np.sum(X[height/2:,width/2:], 1), 0) - u1_bottom_right
        u1 /= X1_count
        u2 /= X2_count
        u3 /= X3_count
        u4 /= X4_count
        u5 /= X5_count
        u[:] = [u1, u2, u3, u4, u5]
        alpha[:] = [X1_count, X2_count, X3_count, X4_count, X5_count]
        alpha /= X_count

    if with_noise:
        for index in range(k):
            u[index] += (random.random() - 0.5) * 0.1 * u[index]

    return u, alpha


# d, k -> sigma
def init_covariance_matrices(d, k):
    sigma = np.empty([k, d, d])
    for index in range(k):
        sigma[index] = np.identity(d, dtype=np.float64)

    return sigma


# X, u, alpha, sigma -> f, p
def calculate_prob_values(X, u, alpha, sigma):
    pass


# alpha, f, Ltab -> Ltab
def add_log_likelihood(alpha, f, Ltab):
    pass


# Ltab -> True|False
def should_stop(Ltab):
    pass


# X, p -> u_new, alpha_new, sigma_new
def update(X, p):
    pass


# log_likelihood, k, d, N -> rating
def calculate_rating(log_likelihood, k, d, N):
    pass


# p -> p_best_regions
def best_regions(p):
    pass
