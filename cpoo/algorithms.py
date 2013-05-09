#!/usr/bin/python
# -*- coding: utf-8 -*-

from copy import deepcopy
from PyQt4 import QtCore
from PyQt4 import QtGui

def thresholding(image):
    size = image.size()
    for i in range(size.width()):
        for j in range(size.height()):
            pixel_color = image.pixel(i, j)
            qcolor = QtGui.QColor(pixel_color)
            mid = (qcolor.red() + qcolor.green() + qcolor.blue()) / 3
            new_color = QtGui.QColor(u'blue') if mid > 100 else QtGui.QColor(u'black')
            image.setPixel(i, j, new_color.rgb())

    return image

def ml_em(image):
    size = image.size()
    for i in range(size.width()):
        for j in range(size.height()):
            pixel_color = image.pixel(i, j)
            qcolor = QtGui.QColor(pixel_color)
            mid = (qcolor.red() + qcolor.green() + qcolor.blue()) / 3
            new_color = QtGui.QColor(u'red') if mid > 100 else QtGui.QColor(u'black')
            image.setPixel(i, j, new_color.rgb())

    return image
