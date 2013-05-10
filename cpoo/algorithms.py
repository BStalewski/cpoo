#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image
import ImageQt
import ImageColor

from copy import deepcopy
from PyQt4 import QtCore
from PyQt4 import QtGui

BLUE = ImageColor.getrgb(u'Blue')
BLACK = ImageColor.getrgb(u'Black')
RED = ImageColor.getrgb(u'Red')


def thresholding(file_name):
    image = Image.open(file_name)
    width, height = image.size
    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j))
            avg_color = sum(pixel[:3]) / 3
            new_color = BLUE if avg_color > 100 else BLACK
            new_pixel = new_color + pixel[3:]
            image.putpixel((i, j), new_pixel)

    return image


def ml_em(file_name):
    image = Image.open(file_name)
    width, height = image.size
    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j))
            avg_color = sum(pixel[:3]) / 3
            new_color = RED if avg_color > 100 else BLACK
            new_pixel = new_color + pixel[3:]
            image.putpixel((i, j), new_pixel)

    return image

