#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image

import math
import numpy as np
import os
import random

from datetime import datetime


'''
Oznaczenia (spojne w calym pliku):
* image - obraz
* width, height - szerokość przekształcanego obrazu
* d - liczba cech (d = 4)
* k - aktualna liczba klastrów (od 2 do K)
* K - maksymalna liczba klastrów (K = 5)
* X - cechy obrazu; wymiary: [height][width][d]
    X[h][w] - wektor cech piksela w współrzędnych (w, h)
    X[h][w][i] - i-ta cecha piksela o współrzędnych (w, h)
* u - wartości średnie; wymiary: [i][d];
    u[i] - wektor wartości średnich każdej cechy i-tego
    klastra, u[i][d] - wartość średnia d-tej cechy w i-tym klastrze
* alpha - wektor wag klastrów, wymiary: [k]
    alpha[i] - waga i-tego klastra
    waga jest zależna od dopasowania punktów do parametrów klastra
* sigma - macierze kowariancji; wymiary: [k][d][d]
    sigma[i] - macierz kowariancji dla cech (d cech) i-tego klastra
* p - prawdopodobieństwa przynależności punktów do klastrów; wymiary [k][height][width]
    p[i] - macierz określająca prawdopodobieństwo przynależności pikseli do i-tego klastra
    p[i][h][w] - prawdopodobieństwo przynależności punktu o współrzędnych (w, h) do i-tego klastra
    dla każdego h, w: suma dla i=0..(k-1) p[i][h][w] = 1.
* f - funkcja gęstości prawdopodobieństwa przynależności punktów do klastrów; wymiary [k][height][width]
    f[i] - funkcja gęstości prawodpodobieństwa przynależności punktów do i-tego klastra
    f[i][h][w] - funkcja gęstości prawodpodobieństwa przynależności punktu o współrzędnych (w, h)
    do i-tego klastra
* f_total - zsumowana funkcja gęstości prawodpodobieństwa dla wszystkich klastrów
* L - prawdopodobieństwo (likelihood) poprawnego dopasowania punktów do klastrów
* Ltab - tablica zawierająca obliczone wartości L w poprzednich iteracjach
* p_best_regions - indeksy klastrów mówiące, do którego klastra przynależy dany
    punkt; wymiary: [height][width]
    p_best_regions[h][w] - numer klastra, do którego należy punkt o współrzędnych (w, h),
    numery klastrów mają wartości: 0, 1, ..., k-1.
* fscale - wartość skalująca wartości cech (wartości odczytane z obrazu są
    przemnażane przez fscale, przy niskiej wartości parametru istnieje ryzyko
    wystąpienia przepełnienia lub zbytniego zbliżenia wartości do zera, przez co
    traktowane są jako zera
'''

MIN_ITERATIONS = 5  # minimalna liczba iteracji dla zadanego k
MAX_ITERATIONS = 20  # maksymalna liczba iteracji dla zadanego k
MIN_ACCEPTABLE_GROWTH = 0.01  # minimalny akceptowalny wzrost, ktory
                              # nie powoduje przerwania iteracji algorytmu


def ml_em(file_name, K=5):
    image = Image.open(file_name)
    k = 2
    fscale = 0.1
    now = datetime.now()
    # Inicjalizacja parametrów: cechy X, wartości średnie u,
    # wagi alpha, macierze kowariancji sigma
    X = init_features(image, fscale)
    height, width, d = X.shape
    N = height * width
    u, alpha, sigma = init_distribution_params(X, k, False)
    Ltab = []
    stop = False
    max_iterations = MAX_ITERATIONS
    for i in range(0, max_iterations + 1):
        print 'Iteracja:', i
        # Obliczenie gęstości prawdopodobieństwa, prawdopodobieństw
        # (p oraz L), sprawdzenie, czy warunek stopu jest spełniony
        f, p = calculate_prob_values(X, u, alpha, sigma)
        Ltab = add_log_likelihood(alpha, f, Ltab)
        stop = should_stop(Ltab)
        if stop:
            print 'Stop w iteracji:', i
            break

        # aktualizacja parametrów rozkładu klastrów: u, sigma oraz ich wag alpha
        u, alpha, sigma = update(X, p)
        # wyznaczenie oceny rozkładu
        rating = calculate_rating(Ltab[-1], k, d, N)
        print 'rating:', rating

    p_best_regions = best_regions(p)

    print '========================='
    print 'liczba pikseli obrazu:', N
    print 'liczba pikseli w klastrze nr 0:', N - np.count_nonzero(p_best_regions)
    print 'liczba pikseli w klastrze nr 1:', N - np.count_nonzero(p_best_regions - 1)
    print 'liczba pikseli w klastrze nr 2:', N - np.count_nonzero(p_best_regions - 2)
    print 'liczba pikseli w klastrze nr 3:', N - np.count_nonzero(p_best_regions - 3)
    print 'liczba pikseli w klastrze nr 4:', N - np.count_nonzero(p_best_regions - 4)

    after = datetime.now()
    print 'czas wykonania:', after - now

    # Wyznaczenie bazowej nazwy dla pliku wyjściowego:
    # dla pliku abc.jpg przy i klastrach będzie to abc_ki.jpg, np abc_k3.jpg
    # w funkcji make_result_images nazwa ta zostanie wykorzystana i dla każdego
    # klastra zostanie utworzony plik: abc_k3_i.jpg, gdzie i to numer klastra,
    # czyli będą to pliki abc_k3_0.jpg, abc_k3_1.jpg oraz abc_k3_2.jpg.
    head, tail = os.path.split(file_name)
    tail_parts = tail.rsplit('.')
    tail_parts.insert(1, '_k{k}.'.format(k=k))
    new_tail = ''.join(tail_parts)
    out_file_name = os.path.join(head, new_tail)

    result_image = make_result_images(image, p_best_regions, out_file_name, k)

    return result_image


# image -> X
def init_features(image, fscale):
    ''' Wyznaczenie cech.'''
    width, height = image.size
    X = np.empty([height, width, 4])
    for ind_h in range(height):
        for ind_w in range(width):
            pixel = image.getpixel((ind_w, ind_h))
            r_g = pixel[0] - pixel[1]
            r_b = pixel[0] - pixel[2]
            X[ind_h][ind_w] = [ind_w * fscale, ind_h * fscale, r_g * fscale, r_b * fscale]

    return X


# X, k -> u, alpha, sigma
def init_distribution_params(X, k, with_noise):
    ''' Wyznaczenie początkowych wartości średnich, wag i m. kowariancji.'''
    u, alpha = init_means_weights(X, k, with_noise)
    d = X.shape[2]
    sigma = init_covariance_matrices(d, k)
    return u, alpha, sigma


# X, k -> u, alpha
def init_means_weights(X, k, with_noise):
    ''' Wyznaczenie wartości średnich i wag.
    parametr with_noise - czy do w. średnich ma być dodany losowy szum,
    obecnie nieużywany.'''
    height, width, d = X.shape
    X_count = width * height
    u = np.empty([k, d])
    alpha = np.empty(k)
    # dla każdego k inaczej należy wyznaczyć początkowy klaster w inny sposób
    # strona 6 atrykułu na górze -> opis, jaki wygląd mają klastry
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

    # dodanie losowego szumu (obecnie nieużywane)
    if with_noise:
        for index in range(k):
            u[index] += (random.random() - 0.5) * 0.1 * u[index]

    return u, alpha


# d, k -> sigma
def init_covariance_matrices(d, k):
    ''' Początkowe macierze kowariancji - identyczności.'''
    sigma = np.empty([k, d, d])
    for index in range(k):
        sigma[index] = np.identity(d, dtype=np.float64)

    return sigma


# X, u, alpha, sigma -> f, p
def calculate_prob_values(X, u, alpha, sigma):
    ''' Policzenie prawdopodobieństw i f. gęstości prawdopodobieństwa.'''
    f = calculate_prob_density(X, u, sigma)
    p = calculate_prob(X, alpha, f)
    return f, p


# X, u, sigma -> f
# from: http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
def calculate_prob_density(X, u, sigma):
    ''' Funkcja licząca wartość f. gęstości prawdopodobieństwa.
    Zgodnie ze wzorem z:http://en.wikipedia.org/wiki/Normal_distribution w komórce pdf.'''
    height, width, d = X.shape
    k = u.shape[0]
    f = np.empty([k, height, width])
    for i in range(k):
        if d == len(u[0]) and (d, d) == sigma[0].shape:
            det = np.linalg.det(sigma[i])
            if det == 0:
                print sigma[i]
                raise NameError("The covariance matrix can't be singular")
            try:
                norm_const = 1.0 / (math.pow((2 * np.pi), float(d)/2) * math.pow(det, 1.0 / 2))
            except ValueError:
                v1 = 2 * np.pi
                print 'v1:', v1
                v2 = float(d) / 2
                print 'v2:', v2
                v3 = math.pow(v1, v2)
                print 'v3:', v3
                v4 = 1.0 / v3
                print 'v4:', v4
                print 'sigma[{i}]:'.format(i=i), sigma[i]
                v5 = math.pow(det, 1.0 / 2)
                print 'v5:', v5
                v6 = v4 * v5
                print 'v6:', v6
                raise

            inv = np.matrix(sigma[i]).I
            for ind_h in range(height):
                for ind_w in range(width):
                    x_mu = np.matrix(X[ind_h][ind_w] - u[i])
                    try:
                        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
                    except OverflowError:
                        print 'OVERFLOW_ERROR'
                        print '-0.5 * ... =', -0.5 * (x_mu * inv * x_mu.T)
                        raise
                    f[i][ind_h][ind_w] = result
            f[i] *= norm_const
        else:
            raise NameError("The dimensions of the input don't match")

    return f


# X, alpha, f -> p
def calculate_prob(X, alpha, f):
    ''' Wyznaczenie prawdopodobieństw przynależności pikseli do klastrów.'''
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
    ''' Wyznaczenie i dodanie log likelihood do tablicy Ltab.'''
    f_total = calculate_total_prob_density(alpha, f)
    L = calculate_log_likelihood(f_total)
    Ltab_new = update_log_likelihoods(Ltab, L)
    return Ltab_new


# alpha, f -> f_total
def calculate_total_prob_density(alpha, f):
    ''' Obliczenie zsumowanej f. gęstości prawdopodobieństw.'''
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
    ''' Sprawdzenie warunku stopu: zatrzymanie algorytmu jeśli obliczono minimalną liczbę
    iteracji oraz w ostatnich MIN_ITERATIONS wzrost jakości był mniejszy niż 1%.'''
    if len(Ltab) < MIN_ITERATIONS + 1:
        return False
    else:
        zipped_last_growths = zip(Ltab[-MIN_ITERATIONS:], Ltab[-(MIN_ITERATIONS+1):-1])
        low_growths = [((next - prev) / prev) < MIN_ACCEPTABLE_GROWTH for next, prev in zipped_last_growths]
        return any(low_growths)


# X, p -> u_new, alpha_new, sigma_new
def update(X, p):
    ''' Aktualizacja parametrów rozkładu.'''
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
                # na pierwszy argument (różnica ze wzorem w artykule)
                result = p[i][ind_h][ind_w] * (x_u.T * x_u)
                sigma[i] += result
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
    ''' Wyznaczenie najlepszych klastrów dla punktów obrazka.
    Wynikiem jest macierz wielkości obrazu wejściowego zawierająca indeksy klastrów.'''
    k, height, width = p.shape
    p_best_regions_int32 = np.argmax(p, 0)
    # konwersja, ponieważ później ta macierz będzie wykorzystywana do zapisu
    # obrazu, a wymagany jest format uint8
    p_best_regions_int8 = np.asarray(p_best_regions_int32, dtype=np.uint8)
    p_best_regions = p_best_regions_int8
    return p_best_regions


# image, p_best_regions, file_name, k -> result_image
def make_result_images(image, p_best_regions, file_name, k):
    '''Zapisanie obrazów wyjściowych na dysku. Zostanie zapisanych
    k obrazów, z których każdy zawiera kolorową część -> część należąca do
    danego klastra oraz czarną -> część należącą do innych klastrów.'''
    height, width = p_best_regions.shape
    image_data_array = np.asarray(image) # a is readonly
    tmp_array = np.empty((height, width, image_data_array.shape[2]), dtype=np.uint8)
    for index in range(image_data_array.shape[2]):
        tmp_array[:,:,index] = p_best_regions

    for i in range(k):
        head, tail = os.path.split(file_name)
        tail_parts = tail.rsplit('.')
        tail_parts.insert(1, '_{i}.'.format(i=i))
        new_tail = ''.join(tail_parts)
        out_file_name = os.path.join(head, new_tail)
        mask_array = np.copy(tmp_array)
        mask_array += (1 - i)
        np.place(mask_array, mask_array != 1, 0)
        result_image_data_array = image_data_array * mask_array
        image = Image.fromarray(result_image_data_array)
        image.save(out_file_name)

    return image
