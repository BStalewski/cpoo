#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image
import ImageColor


BLUE = ImageColor.getrgb(u'Blue')
BLACK = ImageColor.getrgb(u'Black')
RED = ImageColor.getrgb(u'Red')


def thresholding(file_name, thresholds_count):
    image = Image.open(file_name)
    width, height = image.size
    threshold_value = thresholds_count * 256 / 20
    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j))
            avg_color = sum(pixel[:3]) / 3
            new_color = BLUE if avg_color > threshold_value else BLACK
            new_pixel = new_color + pixel[3:]
            image.putpixel((i, j), new_pixel)

    return image


def ml_em(file_name, arg1, arg2):
    image = Image.open(file_name)
    width, height = image.size
    threshold_value = arg2 * 256 / 20
    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j))
            avg_color = sum(pixel[:3]) / 3
            new_color = RED if avg_color > threshold_value else BLACK
            new_pixel = new_color + pixel[3:]
            image.putpixel((i, j), new_pixel)

    return image

