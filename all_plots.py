# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:53:34 2017

@author: mildbyte
"""

import itertools
import numpy as np
from matplotlib import pyplot as plt
import PIL.Image
import PIL.ImageDraw
import subprocess

from dataloading import load_cells_npcs, Location
from travel import construct_graph, dijkstra, get_route, get_closest_exterior_location
from drawing import draw_route, mapspace_small, to_map, to_world
from population import draw_npcs

MAP_PATH = "map_small.jpg"


# Load stuff from the Enchanted Editor dump
cells, npcs = load_cells_npcs("../Morrowind.esm.txt")

npc_ids = [n.name for n in npcs]
npc_locations = {n.name : Location(n.position, cell=c) for c in cells for n in c.references if n.name in npc_ids}

aryon = npc_locations['aryon']
vertices, edges = construct_graph(npcs, cells, [aryon])


# Calculate the best travel paths from Aryon's location to all locations
# in the world
dist, prev = dijkstra(vertices, edges, aryon)


# Print out an example route
target = [v for v in vertices if 'Hlormaren, Dome' in v.cell.name][0]
route = get_route(prev, target)
for r in route:
    print r


# Test displaying points on map
map_orig = PIL.Image.open(MAP_PATH)
drawer = PIL.ImageDraw.Draw(map_orig)

for point in mapspace_small.T:
    drawer.ellipse((point[0]-20,point[1]-20,point[0]+20,point[1]+20), fill=(128,0,0))
    
arr = get_closest_exterior_location(aryon).coords
ar_m = to_map(arr[0], arr[1])
drawer.ellipse((ar_m[0]-20,ar_m[1]-20,ar_m[0]+20,ar_m[1]+20), fill=(0,0,128))

map_orig.save("map_small_aryon_location.png")


# And that the pathfinding makes sense
draw_route(prev, target, MAP_PATH, "map_aryon_hlormaren.png")


# We want to find out the time Aryon needs to travel to each pixel
# on the game map
ext_loc, ext_dist = zip(*((l.coords, d) for l, d in dist.iteritems() if not l.cell.is_interior))

# Time Aryon needs to travel to each exterior location
with open('locations.txt', 'w') as f:
    f.write('\n'.join("%f %f %f" % (l[0], l[1], d) for l, d in zip(ext_loc, ext_dist)))
    
# All pixels on the map with the in-game coordinates
with open('sought.txt', 'w') as f:
    for x, y in itertools.product(xrange(map_orig.size[0]), xrange(map_orig.size[1])):
        xw, yw, _ = to_world(x + 0.5, y + 0.5)
        f.write("%f %f\n" % (xw, yw))

# Run up a C program that outputs the travel time (smallest of times to
# travel to each exterior location and from there to the target point) to
# each sought pixel.
subprocess.check_call('./get_travel_times.exe')

# Read its output back
map_orig = PIL.Image.open(MAP_PATH)
overlay = np.zeros(map_orig.size)
with open("result.txt") as f:
    for (x, y), d in zip(itertools.product(xrange(map_orig.size[0]), xrange(map_orig.size[1])), f.readlines()):
        overlay[x, y] = float(d)

plt.imshow(map_orig, alpha=0.5)
CS = plt.contour(overlay.T, alpha=1, linewidths=3)
plt.clabel(CS, inline=1, fontsize=10, fmt=lambda c: "%.2fh" % c)
plt.savefig("map_aryon_travel_contour.png") # doesn't work well, had to screenshot instead


# Plot some population heatmaps

draw_npcs(npcs, npc_locations, MAP_PATH, 'map_population.png',
          filter_sigma=25, sigmoid_k=8, sigmoid_c=0.2) # all NPCs

draw_npcs(npcs, npc_locations, MAP_PATH, 'map_population_darkelf.png',
          mark_npcs=[n for n in npcs if n.race == 'Dark Elf'],
          filter_sigma=25, sigmoid_k=8, sigmoid_c=0.2) # only Dark Elves

draw_npcs(npcs, npc_locations, MAP_PATH, 'map_population_darkelf_relative.png',
          mark_npcs=[n for n in npcs if n.race == 'Dark Elf'], relative=True,
          filter_sigma=50, sigmoid_k=4, sigmoid_c=0.5) # Dark Elf fraction

draw_npcs(npcs, npc_locations, MAP_PATH, 'map_population_slave_relative.png',
          mark_npcs=[n for n in npcs if n.class_name == 'Slave'], relative=True,
          filter_sigma=25, sigmoid_k=8, sigmoid_c=0.2) # Slave fraction

draw_npcs(npcs, npc_locations, MAP_PATH, 'map_population_female_relative.png',
          mark_npcs=[n for n in npcs if n.is_female], relative=True,
          filter_sigma=50, sigmoid_k=12, sigmoid_c=0.7) # Female fraction
