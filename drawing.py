# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:11:03 2017

@author: mildbyte

Random functions for drawing stuff on the in-game map
"""

import numpy as np
import scipy.linalg
import PIL.Image
import PIL.ImageDraw

from travel import get_closest_exterior_location, get_route


worldspace = np.array([[-12353, 57603, 1],
                       [32862.45, -106875.9, 1],
                       [53126.93, 153595.4, 1]]).T

# Coordinates on a large Morrowind map that includes the landmass created by
# Tamriel Rebuilt
mapspace = np.array([[1124, 654, 1],
                    [1282, 1227, 1],
                    [1350, 320, 1]]).T

# Coordinates on the small Morrowind map that only has the original in-game island
mapspace_small = np.array([[619, 880, 1],
                    [865, 1777, 1],
                    [971, 360, 1]]).T

W = mapspace_small.dot(scipy.linalg.inv(worldspace))
M = np.linalg.inv(W)


def to_world(x, y):
    trans = M.dot([float(x), float(y), 1.])
    return (trans[0], trans[1], 0)


def to_map(x, y):
    trans = W.dot([float(x), float(y), 1.])
    return (trans[0], trans[1])


def draw_route(prev, dest, map_path, output_path):
    # Given a dictionary of location -> location produced by Dijkstra,
    # plots the route from the source to a given destination.
    map_orig = PIL.Image.open(map_path)
    drawer = PIL.ImageDraw.Draw(map_orig)
    
    route = get_route(prev, dest)
    
    coords = []    
    
    for r in route:
        c = get_closest_exterior_location(r).coords
        transf = to_map(c[0], c[1])
        coords.append((int(round(transf[0])), int(round(transf[1]))))
        drawer.ellipse((transf[0]-10,
                        transf[1]-10,
                        transf[0]+10,
                        transf[1]+10), fill=(0,128,0))
    
    drawer.line(coords, fill=128, width=5)
    
    map_orig.save(output_path)