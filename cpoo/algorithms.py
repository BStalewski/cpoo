#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image
import ImageColor
import otsu
import ml_em

BLUE = ImageColor.getrgb(u'Blue')
BLACK = ImageColor.getrgb(u'Black')
RED = ImageColor.getrgb(u'Red')


def thresholding(file_name, thresholds_count):
    return otsu.otsu(file_name, thresholds_count)


def ml_em_thresholding(file_name, arg1, arg2):
    return ml_em.ml_em(file_name)
    #return otsu.otsu(file_name, int(arg1))
