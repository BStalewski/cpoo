#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image
import ImageColor
import random

import math
import numpy as np

from datetime import datetime


BLUE = ImageColor.getrgb(u'Blue')
BLACK = ImageColor.getrgb(u'Black')
RED = ImageColor.getrgb(u'Red')


def ml_em(file_name, K=5):
    image = Image.open(file_name)
    k = 2
    fscale = 10
    now = datetime.now()
    X = init_features(image, fscale)
    height, width, d = X.shape
    N = height * width
    u, alpha, sigma = init_distribution_params(X, k, False)
    Ltab = []
    print '1111111111111111111111111111'
    print 'u:', u
    print 'alpha:', alpha
    print 'sigma:', sigma
    print '2222222222222222222222222222'
    f, p = calculate_prob_values(X, u, alpha, sigma)
    print '3333333333333333333333333333'
    Ltab = add_log_likelihood(alpha, f, Ltab)
    print '4444444444444444444444444444'
    stop = should_stop(Ltab)
    print 'stop?:', stop
    after = datetime.now()
    print 'f:', f
    print 'fmax:', f.max()
    print 'fmin:', f.min()
    print 'f[0][79][80]:', f[0][79][80]
    print 'npm[0][79][80]:', norm_pdf_multivariate(X[79][80], u[0], np.matrix(sigma[0]))
    print 'f[1][79][80]:', f[1][79][80]
    print 'npm[1][79][80]:', norm_pdf_multivariate(X[79][80], u[1], np.matrix(sigma[1]))
    print 'f[0][0][0]:', f[0][0][0]
    print 'npm[0][0][0]:', norm_pdf_multivariate(X[0][0], u[0], np.matrix(sigma[0]))
    print p
    p_sum = np.sum(p, 0)
    print 'np.max(p_sum):', np.max(p_sum)
    print 'np.min(p_sum):', np.min(p_sum)
    print '5555555555555555555555555555'
    u, alpha, sigma = update(X, p)
    print 'After update:'
    print 'u:', u
    print 'alpha:', alpha
    print 'sigma:', sigma
    print '6666666666666666666666666666'
    rating = calculate_rating(Ltab[-1], k, d, N)
    print 'rating:', rating

    print 'diff:', after - now
    return image


# image -> X
def init_features(image, fscale, d=4):
    width, height = image.size
    X = np.empty([height, width, d])
    for ind_h in range(height):
        for ind_w in range(width):
            pixel = image.getpixel((ind_w, ind_h))
            r_g = pixel[0] - pixel[1]
            r_b = pixel[0] - pixel[2]
            X[ind_h][ind_w] = [ind_w / fscale, ind_h / fscale, r_g / fscale, r_b / fscale]

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
    f = calculate_prob_density(X, u, sigma)
    p = calculate_prob(X, alpha, f)
    return f, p


# X, u, sigma -> f
# from: http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
def calculate_prob_density(X, u, sigma):
    height, width, d = X.shape
    K = u.shape[0]
    f = np.empty([K, height, width])
    for i in range(K):
        if d == len(u[0]) and (d, d) == sigma[0].shape:
            det = np.linalg.det(sigma[i])
            if det == 0:
                raise NameError("The covariance matrix can't be singular")
            norm_const = 1.0 / (math.pow((2 * np.pi), float(d)/2) * math.pow(det, 1.0 / 2))
            inv = np.matrix(sigma[i]).I
            if i == 0:
                print 'det:', det
                print 'norm_const:', norm_const
                print 'inv:', inv
            for ind_h in range(height):
                for ind_w in range(width):
                    x_mu = np.matrix(X[ind_h][ind_w] - u[i])
                    result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
                    if i == 0 and ind_h == 0 and ind_w == 0:
                        print 'x_mu:', x_mu
                        print 'pow:', -0.5 * (x_mu * inv * x_mu.T)
                        print 'result:', result
                    f[i][ind_h][ind_w] = result
            f[i] *= norm_const
        else:
            raise NameError("The dimensions of the input don't match")

    return f


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0 / (math.pow((2 * np.pi), float(size)/2) * math.pow(det, 1.0 / 2))
        x_mu = np.matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


# X, alpha, f -> p
def calculate_prob(X, alpha, f):
    height, width, d = X.shape
    k = alpha.shape[0]
    p = np.empty([k, height, width])
    for ind_h in range(height):
        for ind_w in range(width):
            p[:, ind_h, ind_w] = alpha * f[:, ind_h, ind_w]
            p[:, ind_h, ind_w] /= np.sum(p[:, ind_h, ind_w])

    return p


# alpha, f, Ltab -> Ltab_new
def add_log_likelihood(alpha, f, Ltab):
    before = datetime.now()
    f_total = calculate_total_prob_density(alpha, f)
    after = datetime.now()
    print 'f_total:', f_total
    print 'test time:', after - before
    L = calculate_log_likelihood(f_total)
    print 'L:', L
    Ltab_new = update_log_likelihoods(Ltab, L)
    print 'Ltab -> Ltab_new', Ltab, '->', Ltab_new
    return Ltab_new


# alpha, f -> f_total
def calculate_total_prob_density(alpha, f):
    alpha_tab = np.empty([len(alpha), 1, 1])
    alpha_tab[:,0,0] = alpha
    f_total = np.sum(np.multiply(f, alpha_tab), 0)

    return f_total


# f_total -> L
def calculate_log_likelihood(f_total):
    L = np.sum(np.log(f_total))
    return L


# Ltab, L -> Ltab_new
def update_log_likelihoods(Ltab, L):
    Ltab_new = Ltab[:]
    Ltab_new.append(L)
    return Ltab_new


# Ltab -> True|False
def should_stop(Ltab):
    if len(Ltab) < 11:
        return False
    else:
        zipped_last_growths = zip(Ltab[-10:], Ltab[-11:-1])
        low_growths = [((next - prev) / prev) < 0.01 for next, prev in zipped_last_growths]
        return any(low_growths)


# X, p -> u_new, alpha_new, sigma_new
def update(X, p):
    u_new = update_means(X, p)
    alpha_new = update_weights(p)
    sigma_new = update_covariance_matrices(X, u_new, p)
    return u_new, alpha_new, sigma_new


# X, p -> u
def update_means(X, p):
    height, width, d = X.shape
    k, height, width = p.shape
    u = np.empty([k, d])
    for i in range(k):
        for j in range(d):
            u[i, j] = np.sum(np.sum(X[:,:,j] * p[i], 1), 0)
        u[i] /= np.sum(np.sum(X, 1), 0)

    return u


# p -> alpha
def update_weights(p):
    k, height, width = p.shape
    total = height * width
    alpha = np.empty(k)
    alpha = np.sum(np.sum(p, 2), 1) / total
    return alpha


# X, u, p -> sigma
def update_covariance_matrices(X, u, p):
    height, width, d = X.shape
    k, height, width = p.shape
    sigma = np.empty([k, d, d])
    for i in range(k):
        p_sum = np.sum(p[i])
        for ind_h in range(height):
            for ind_w in range(width):
                x = X[ind_h][ind_w]
                x_u = np.matrix(x - u[i])
                # ze wzgledu na reprezentacje, zmiana transpozycji
                # na pierwszy argument
                result = p[i][ind_h][ind_w] * (x_u.T * x_u)
                sigma[i] += result
        print 'sigma[i].shape:', sigma[i].shape
        print 'p_sum.shape:', p_sum.shape
        sigma[i] /= p_sum

    return sigma


# log_likelihood, k, d, N -> rating
def calculate_rating(log_likelihood, k, d, N):
    mk = calculate_mk(k, d)
    rating = log_likelihood - mk / 2.0 * math.log(N)
    return rating


# k, d -> mk
def calculate_mk(k, d):
    mk = (k - 1.0) + k * d + k * d * (d + 1) / 2.0
    return mk


# p -> p_best_regions
def best_regions(p):
    pass
