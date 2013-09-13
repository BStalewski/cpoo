#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image
import ImageColor

import math
from numpy import *


BLUE = ImageColor.getrgb(u'Blue')
BLACK = ImageColor.getrgb(u'Black')
RED = ImageColor.getrgb(u'Red')


def ml_em(file_name):
    image = Image.open(file_name)
    means, covs, alphas, features = divide_to_windows(image)

    print features
    print alphas
    should_stop = False
    while not should_stop:
        means, covs, alphas = update_params(image, means, covs, alphas, features)
        should_stop = update_stop_param(image, means, covs, alphas, features)


    '''
    width, height = image.size
    threshold_value = arg2 * 256 / 20
    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j))
            avg_color = sum(pixel[:3]) / 3
            new_color = RED if avg_color > threshold_value else BLACK
            new_pixel = new_color + pixel[3:]
            image.putpixel((i, j), new_pixel)
    '''

    return image


def get_features(image):
    features = []
    for i in range(width):
        features.append([])
        for j in range(height):
            pixel = image.getpixel((i, j))
            r_g = pixel[0] - pixel[1]
            r_b = pixel[0] - pixel[2]
            features[i].append([i, j, r_g, r_b])

    return features


def divide_to_windows(image, count=2):
    width, height = image.size
    means = []
    alphas = []

    features = get_features(image)
    features_count = len(features)

    means1 = [0.0, 0.0, 0.0, 0.0]
    means2 = [0.0, 0.0, 0.0, 0.0]
    covs = []
    w1_width = width / 2
    w1_height = height / 2
    w_total = width * height
    for i in range(width / 4, 3 * width / 4):
        for j in range(height / 4, 3 * height / 4):
            for k in range(features_count):
                means1[k] += features[i][j][k]

    w1_total = w1_width * w1_height
    means1 = [val / w1_total for val in means1]
    alpha1 = float(w1_total) / w_total
    means.append(means1)
    alphas.append(alpha1)

    for i in range(width):
        for j in range(height):
            if width / 4 < i < 3 * width / 4 or height / 4 < j < 3 * height / 4:
                continue

            for k in range(features_count):
                means2[k] += features[i][j][k]

    w2_total = width * height - w1_total
    means2 = [val / w2_total for val in means2]
    alpha2 = float(w2_total) / w_total
    means.append(means2)
    alphas.append(alpha2)

    for _ in range(count):
        covs.append([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    return means, covs, alphas, features


def update_params(image, means, covs, alphas, features):
    windows_count = len(means)
    p_matrices = [calculate_p_matrix(features, k, means, covs, alphas) for k in range(windows_count)]
    new_alphas = [0.0 for _ in range(windows_count)]
    width = len(features)
    height = len(features[0])
    total = width * height
    for k in range(windows_count):
        for i in range(width):
            for j in range(height):
                new_alphas[k] += features[i][j][k]
        new_alphas[k] /= total

        for i in range(width):
            for j in range(height):



def update_stop_param(image, means, covs, alphas, features):
    return True


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = numpy.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0 / (math.pow((2 * numpy.pi), float(size)/2) * math.pow(det, 1.0 / 2))
        x_mu = numpy.matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


def calculate_p(x, k, means, covs, alphas):
    p_sum = 0.0
    pk = norm_pdf_multivariate(x, means[k], covs[k])
    for j in range(len(means)):
        p_sum += norm_pdf_multivariate(x, means[j], covs[j])

    return pk / p_sum


def calculate_p_matrix(features, k, means, covs, alphas):
    p_matrix = []
    for i in range(len(features)):
        p_matrix.append([])
        for j in range(len(features[i])):
            p_matrix[-1].append(calculate_p(features[i][j], k, means, covs, alphas))

    return p_matrix
