# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:07:47 2017

@author: mildbyte
"""

import matplotlib.cm
import numpy as np
import PIL.Image
from scipy.ndimage.filters import gaussian_filter

from travel import get_closest_exterior_location
from drawing import to_map


def sigmoid(t, k=8, c=0.1):
    return 1./(1 + np.exp(k*(c-t)))


def get_overlay_for(npcs, npc_locations, size):
    # For each pixel on the map, find out how many NPCs are the closest to this
    # pixel (in the interior or the exterior)
    overlay = np.zeros(size)
    for n in npcs:
        if n.name not in npc_locations:
            continue
        try:
            coord = get_closest_exterior_location(npc_locations[n.name]).coords
        except AssertionError:
            continue
        
        x, y = to_map(coord[0], coord[1])
        overlay[x, y] += 1
    
    return overlay


def draw_npcs(all_npcs, npc_locations, input_path, output_path, mark_npcs=None,
              relative=False, filter_sigma=10, sigmoid_k=8, sigmoid_c=0.1):

    map_orig = PIL.Image.open(input_path)
    
    # Get the overlay for NPCs we wish to plot
    if not mark_npcs:
        mark_npcs = all_npcs    
    overlay = get_overlay_for(mark_npcs, npc_locations, map_orig.size)
    
    # If we want a relative fraction, we also want an overlay for the entire population
    if relative:
        overlay_population = get_overlay_for(all_npcs, npc_locations, map_orig.size)
        overlay = np.nan_to_num(np.divide(overlay, gaussian_filter(overlay_population, filter_sigma)))
    
    # Blur and normalize it (TODO if relative, we've already blurred the overall
    # population overlay once, is that a problem?)
    overlay = gaussian_filter(overlay, filter_sigma)
    overlay /= np.max(overlay)
    overlay = sigmoid(overlay, k=sigmoid_k, c=sigmoid_c)
    
    # Apply a colormap and delete the alpha channel
    overlay_cm = np.delete(matplotlib.cm.Blues(overlay.T), 3, axis=2)
    overlay_im = PIL.Image.fromarray(np.uint8(overlay_cm * 255))
    
    map_final = PIL.Image.blend(map_orig, overlay_im, 0.7)
    map_final.save(output_path) 