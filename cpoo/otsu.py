#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image
import ImageColor
import os
import numpy as np
from matplotlib import pyplot as plt

NGRAY = 256;

def otsu(file_name, thresholds_count):
    #Convert_to_grayscale(image)
    image = get_grayscaled(file_name);
    width, height = image.size
    MLEVEL = thresholds_count + 1;
    thresholds =  [ 0 for i in range(MLEVEL) ]
    h =  image.histogram()
    #Build lookup tables from h
    P = [[0 for i in xrange(NGRAY)] for i in xrange(NGRAY)]
    S = [[0 for i in xrange(NGRAY)] for i in xrange(NGRAY)]
    H = [[0 for i in xrange(NGRAY)] for i in xrange(NGRAY)]
    build_lookup_tables(P, S, H, h)
    #Calculate max sigma
    maxSig = find_max_sigma(MLEVEL, H, thresholds)
    msg = 'thresholds: '
    for i in range(0,MLEVEL):
      msg += str(i) + '=' + str(thresholds[i]) + ', '
    print msg + ' maxSig = ' + str(maxSig)
    #Estimate regions
    regions = estimate_regions(MLEVEL, thresholds, list(image.getdata()), width, height)
    save_regions(regions,MLEVEL,file_name)
    #Plot histogram
    plot_histogarm(image,thresholds,file_name)
    return image

def estimate_regions(mlevel,t,pixels,width,height):
    rep = [Image.new("L", [width,height]) for i in xrange(mlevel)]
    for i in range(0,width*height):
        val = 0xff & pixels[i]
        for k in range(0,mlevel):
			if k < mlevel-1:
			  if val < t[k+1] and val > t[k]:
				rep[k].putpixel((i%width, i/width), val)
			else:
			  if val > t[k]:
				rep[k].putpixel((i%width, i/width), val)
    return rep

def save_regions(regions,levels,origin_file):
    head, tail = os.path.split(origin_file)
    for i in range(0,levels):
        file_name = os.path.normpath(head +'/'+ 'reg_' + str(i)+tail)
        regions[i].save(file_name)
        os.startfile(file_name)

def find_max_sigma(maxLevel,H,t):
	tmpThreshold = [ 0 for i in range(maxLevel) ]
	return  find_max_sigma_rec(1,maxLevel,1,0,0, H, tmpThreshold,t)

def find_max_sigma_rec(level,maxLevel,j,sig,mSig,H,tmpT,t):
	maxSig = mSig;
	if level == maxLevel:
		Sq = sig + H[j][255];
		if (maxSig < Sq):
			for k in range(0,maxLevel):
				t[k] = tmpT[k];
			maxSig = Sq;
	else:
		for i in range(j,NGRAY-maxLevel + level):
			tmpT[level] = i;
			maxSig = find_max_sigma_rec(level+1,maxLevel,i+1,sig + H[j][i],maxSig, H,tmpT,t);

	return maxSig;

def build_lookup_tables(P,S,H,h):
    # diagonal
    for i in range(0,NGRAY):
      P[i][i] = h[i]
      S[i][i] = i*h[i]
    #calculate first row (row 0 is all zero)
    for i in range(1,NGRAY-1):
      P[1][i+1] = P[1][i] + h[i+1]
      S[1][i+1] = S[1][i] + (i+1)*h[i+1]
    #using first row to calculate others
    for i in range(2,NGRAY):
      for j in range(i+1,NGRAY):
    	P[i][j] = P[1][j] - P[1][i-1]
        S[i][j] = S[1][j] - S[1][i-1]
    #calculate H[i][j]
    for i in range(1,NGRAY):
      for j in range(i+1,NGRAY):
    	if P[i][j] != 0:
            H[i][j] = (S[i][j]*S[i][j])/P[i][j]
    	else:
	        H[i][j] = 0

def get_grayscaled(file_name):
    image = Image.open(file_name).convert('L')
    head, tail = os.path.split(file_name)
    file_name = os.path.normpath(head +'/'+ 'grayscale_'+tail)
    image.save(file_name)
    image = Image.open(file_name)
    return image

def plot_histogarm(image,thresholds,file_name):
    head, tail = os.path.split(file_name)
    file_name = os.path.normpath(head +'/'+ 'histogram_'+tail)
    hist =  image.histogram()
    plt.plot([ i for i in range(NGRAY) ],hist,'b')
    plt.title('Histogram of ' + tail)
    plt.grid(True)
    max_hist = max(hist)
    for i in thresholds:
        plt.plot([i,i],[0,max_hist], 'r-')
    plt.savefig(file_name);
    plt.show()

