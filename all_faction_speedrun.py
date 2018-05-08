import csv
import subprocess
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import PIL as PIL
import numpy as np
import pydot
from matplotlib import pyplot as plt
from yaml import load

from dataloading import load_cells_npcs, Location
from drawing import to_map
from travel import construct_graph, distance, get_closest_exterior_location

plt.style.use('seaborn')
plt.style.use('seaborn-paper')


def validate(graph):
    for node, contents in graph.items():
        if 'giver' not in contents:
            raise ValueError("Node %s has no giver!" % node)
        for item in contents:
            if item not in ['giver', 'prerequisites', 'description']:
                raise ValueError("Node %s has unknown item %s!" % (node, item))
        for p in contents.get('prerequisites', []):
            if p not in graph:
                raise ValueError("Node %s requires unknown node %s!" % (node, p))


def linearize(graph, start):
    result = start[:]
    visited = set(start)

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        for p in graph[node].get('prerequisites', []):
            visit(p)
        result.append(node)

    while len(visited) != len(graph):
        for n in graph:
            if n not in visited:
                visit(n)
                break
    return result


with open('quest_graph.yaml') as f:
    graph = load(f)
final_nodes = graph['ALL']
start = graph['START']
del graph['ALL']
del graph['START']
validate(graph)

# inverted_graph = defaultdict(list)
# for node, items in graph.items():
#     for p in items.get('prerequisites', []):
#         inverted_graph[p].append(node)
#
# test = linearize(graph, start)
# pprint(test)

# edges = [(p, n) for n in graph for p in graph[n].get('prerequisites', [])]
#
# G = pydot.graph_from_edges(edge_list=edges, directed=True)
#
# with open("quest_graph.png", 'wb') as f:
#     f.write(G.create_png(prog=['sfdp', '-Goverlap=false', '-Gsplines=true']))

# Load stuff from the Enchanted Editor dump
cells, npcs = load_cells_npcs("../Morrowind.esm.txt")

all_node_ids = set([i['giver'].lower() for i in graph.values()])
all_node_locations = {n.name.lower(): Location(n.position, cell=c) for c in cells for n in c.references if n.name.lower() in all_node_ids}

# Make sure we've located all POIs in the speedrun
for n in all_node_ids:
    if n not in all_node_locations:
        print("Failed to locate %s" % n)

walking_speed = 750  # game units per real second assuming levitation + boots of blinding speed
travel_time = 0  # roughly 10 seconds to travel with Silt Strider/Guild guide -- to account for dialogue clicking and loading time + nudge
# the optimiser into not flailing all over the game map

vertices, edges = construct_graph(npcs, cells, extra_locations=all_node_locations.values(), use_realtime_metrics=True, instant_travel_time=travel_time)
sorted_vertices = sorted(vertices)
vertex_index = {v: i for i, v in enumerate(sorted_vertices)}


# Add a Recall edge to e.g. a Mages Guild teleporter
# Assuming it's easy for us to set up a Mark at one (e.g. during the initial stage or when the Silent Pilgrimage quest ends)
def add_recall_edges(vertices, teleport_edges, mg_location, travel_time=0.):
    new_edges = deepcopy(teleport_edges)
    for v in vertices:
        new_edges[v][mg_location] = ('Recall', travel_time)
    return new_edges

recall_edges = add_recall_edges(vertices, edges, all_node_locations['ajira'], travel_time=travel_time)
no_recall_edges = edges

# Farm Floyd-Warshall pathfinding out to a separate C++ program
def export_fw_data(edges, fd):
    INFINITY = 1e10

    for v1 in sorted_vertices:
        for v2 in sorted_vertices:
            d = distance(v1, v2, edges, walking_speed)
            fd.write('%f ' % (d if d is not None else INFINITY))
        fd.write('\n')

# with open("dist.txt", 'wb') as f:
#     export_fw_data(edges, f)
with open("dist_recall.txt", 'wb') as f:
    export_fw_data(recall_edges, f)


# Reconstruct path from the FW dump (strictly speaking prev[i, j] should be called next here,
# but next is a reserved keyword..
def floyd_warshall_path(v1, v2, prev):
    i1, i2 = vertex_index[v1], vertex_index[v2]

    if prev[i1, i2] is None:
        return []
    result = [sorted_vertices[i1]]
    while i1 != i2:
        i1 = prev[i1, i2]
        result.append(sorted_vertices[i1])
    return result


# Do the pathfinding
fw_dist_recall = subprocess.Popen(["./floyd_warshall.exe", "dist_recall.txt", "dist_recall_res.txt", "prev_recall_res.txt"])
fw_dist_recall.wait()

dist_recall = np.loadtxt('dist_recall_res.txt')
prev_recall = np.loadtxt('prev_recall_res.txt', dtype=int)


def floyd_warshall_distance(v1, v2, dist):
    return dist[vertex_index[v1], vertex_index[v2]]


print(floyd_warshall_distance(all_node_locations['vivec_god'], all_node_locations['trebonius artorius'], dist))
print(floyd_warshall_path(all_node_locations['vivec_god'], all_node_locations['trebonius artorius'], prev))

print(floyd_warshall_distance(all_node_locations['vivec_god'], all_node_locations['trebonius artorius'], dist_recall))
print(floyd_warshall_path(all_node_locations['vivec_god'], all_node_locations['trebonius artorius'], prev_recall))


# create a matrix of node-to-node best distances
def make_node_distance_matrix(dist):
    node_distances = defaultdict(dict)
    for n1 in all_node_ids:
        for n2 in all_node_ids:
            node_distances[n1][n2] = floyd_warshall_distance(all_node_locations[n1], all_node_locations[n2], dist) if n1 != n2 else 0
    return node_distances


node_distances = make_node_distance_matrix(dist_recall)

#
# # plot a heatmap of travel times
# node_matrix = []
# for k, v in node_distances.iteritems():
#     node_matrix.append([v[i] for i in node_distances])
# import seaborn as sns
#
# ax = sns.clustermap(node_matrix, yticklabels=node_distances.keys(), xticklabels=node_distances.keys(), figsize=(15, 15))
# sns.set(font_scale=0.7)
# plt.setp(ax.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# plt.setp(ax.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# plt.savefig("travel_heatmap_cluster.png", dpi=200)
# plt.close()


# Export distances and graph constraints for the optimiser
def export_optimiser_dists(node_distances, graph, f):
    all_nodes = sorted(graph.keys())
    for n1 in all_nodes:
        for n2 in all_nodes:
            f.write(str(node_distances[graph[n1]['giver'].lower()][graph[n2]['giver'].lower()]) + ' ')
        f.write('\n')


def export_optimiser_constraints(graph, f):
    all_nodes = sorted(graph.keys())
    node_indices = {n: i for i, n in enumerate(all_nodes)}
    for n1 in all_nodes:
        for n2 in graph[n1].get('prerequisites', []):
            f.write("%d " % node_indices[n2])
        f.write('\n')


with open("dist_graph_recall.txt", 'w') as f:
    export_optimiser_dists(node_distances, graph, f)
with open("dep_graph.txt", 'w') as f:
    export_optimiser_constraints(graph, f)



all_nodes = sorted(graph.keys())
# We skip the Sanctus Shrine quest now so the optimiser can use the same graph all the time
args = [0, 0, all_nodes.index('mg_ajira_1'), all_nodes.index('mg_edwinna_1')]
print(args)

subprocess.check_call(["./ga_optimiser.exe", "dep_graph.txt", "dist_graph_recall.txt", "dist_graph_recall.txt",
                       10000,  # pool size
                       200,  # number of iterations
                       500,  # branching factor
                       5, ]  # number of random swaps to generate a new route
                      + args)


def load_optimiser_pool(graph, f):
    all_nodes = sorted(graph.keys())
    pool = []
    for line in f.readlines():
        pool.append([all_nodes[int(i)] for i in line.split()])
    return pool


with open("optimiser_result.txt", 'r') as f:
    pool = load_optimiser_pool(graph, f)


def evaluate_route(route):
    return sum(node_distances[graph[r1]['giver'].lower()][graph[r2]['giver'].lower()] for r1, r2 in zip(route, route[1:]))

best = pool[0] # pool already exported pre-sorted


def route_valid(route, start):
    if route[:len(start)] != start:
        return False
    completed = set()
    for r in route:
        for p in graph[r].get('prerequisites', []):
            if p not in completed:
                print ("Route not valid: %s requires %s" % (r, p))
                return False
        completed.add(r)
    return True

def get_node_distance(n1, n2):
    return node_distances[graph[n1]['giver'].lower()][graph[n2]['giver'].lower()]

def score_route(route):
    return sum(get_node_distance(n1, n2) for n1, n2 in zip(route, route[1:]))

all_nodes = sorted(graph.keys())
with open("initial_route.txt", 'wb') as f:
    f.write(' '.join(str(all_nodes.index(n)) for n in best) + '\n')

def split_route(route, segments=12):
    # splits route into equally-sized segments
    route_len = score_route(route)
    segment_size = route_len / segments

    result_segments = []

    current_segment_size = 0.
    current_segment = [route[0]]
    for r in route[1:]:
        current_segment.append(r)
        current_segment_size += get_node_distance(current_segment[-1], current_segment[-2])
        if current_segment_size > segment_size:
            result_segments.append(current_segment)
            current_segment_size = 0
            current_segment = [current_segment[-1]]

    result_segments.append(current_segment)
    return result_segments



def get_best_mark_position(route):
    return min(
        # can't use the mark until we've placed it
        (sum(get_node_distance(r1, r2) for r1, r2 in zip(route[:i], route[1:i]))
         + sum(
            # after placing the mark, we have a choice of recalling to it and going to the next node
            # or going to the next node directly
            min(get_node_distance(r, r2), get_node_distance(r1, r2)) for r1, r2 in zip(route[i:], route[i + 1:])),
         i, r) for i, r in enumerate(route)
    )


# reconstruct route and output it

def export_route(route, edges, alternative_edges, alternative_edges_start, alternative_edges_end, prev, alternative_prev, fd):
    alternative_recall = False

    def get_edges():
        return alternative_edges if alternative_recall else edges

    def get_prev():
        return alternative_prev if alternative_recall else prev

    # Resulting route
    writer = csv.writer(fd)
    writer.writerow(['Route', 'Location', 'Description', 'Graph Node'])

    output = []

    for node1, node2 in zip(route, route[1:]):
        if node1 == alternative_edges_start:
            alternative_recall = True
        if node1 == alternative_edges_end:
            alternative_recall = False

        r1_vertex = all_node_locations[graph[node1]['giver'].lower()]
        r2_vertex = all_node_locations[graph[node2]['giver'].lower()]
        graph_node = graph[node1]
        subroute = floyd_warshall_path(r1_vertex, r2_vertex, get_prev())

        output.append([str(subroute[0].cell) if subroute else '', graph_node.get('giver'), graph_node.get('description', node1), node1])

        if len(subroute) > 1:
            for i, r in enumerate(zip(subroute, subroute[1:])):
                r1, r2 = r
                try:
                    method, _ = get_edges()[r1][r2]
                except KeyError:
                    method = "Walk/Fly"

                output.append(['%s -> %s' % (method, r2.cell), '', '', ''])

    graph_node = graph[route[-1]]
    output.append([str(subroute[-1].cell) if subroute else '', graph_node.get('giver'), graph_node.get('description', node2), node2])

    for row in output:
        writer.writerow(row)

with open("route_1mark_v11.csv", 'wb') as f:
    export_route(best, recall_edges, recall_edges, 'mg_ajira_1', 'mg_edwinna_1', prev_recall, prev_recall, f)


# Draw the average time it takes to travel from any vertex on the travel graph to any questgiver
dist_norecall = np.loadtxt('dist_norecall_res.txt')
MAP_PATH = "../map_small.jpg"

def get_average_distance(l, dist):
    distances = 0.
    for nv in graph.values():
        distances += floyd_warshall_distance(l, all_node_locations[nv['giver'].lower()], dist)
    return distances / len(graph)

map_orig = PIL.Image.open(MAP_PATH)
overlay = np.zeros(map_orig.size)

xs = []
ys = []

for l in vertex_index:
    if not l.cell.is_interior:
        map_x, map_y = to_map(l.coords[0], l.coords[1])
        overlay[int(round(map_x)), int(round(map_y))] = get_average_distance(l, dist_norecall)
        xs.append([map_x, map_y])
        ys.append(get_average_distance(l, dist_norecall))

# this doesn't look good, sort of like a Voronoi diagram -- really want to recalculate the travel time from _every_ pixel
# on the map for a good overlay (since we can always Almsivi/Divine/Recall from any point).
from scipy.interpolate import NearestNDInterpolator
nd = NearestNDInterpolator(xs, ys)

overlay = np.array([[nd(i, j) for j in xrange(map_orig.size[1])] for i in xrange(map_orig.size[0])])

plt.figure(figsize=(10, 10))
plt.imshow(map_orig, alpha=0.8)
CS = plt.contour(overlay.T, np.linspace(20, 35, 16), alpha=1, linewidths=2, cmap=plt.get_cmap('winter'))
plt.clabel(CS, inline=1, fontsize=10, fmt=lambda c: "%.2fs" % c)
plt.gca().grid(False)
plt.savefig("map_average_travel.png", dpi=200)



def draw_route(route, edges, alternative_edges, alternative_edges_start, alternative_edges_end, prev, alternative_prev, output_path):
    # todo unify this with export_route?
    alternative_recall = False

    color_map = {
        'Walk/Fly': (200, 200, 200),
        'Door': (200, 200, 200),
        'Almsivi Intervention': (0, 0, 200),
        'Divine Intervention': (0, 0, 200),
        'Guild Guide': (200, 0, 0),
        'Recall': (0, 200, 0),
        'Travel': (200, 200, 0),
    }


    def get_edges():
        return alternative_edges if alternative_recall else edges

    def get_prev():
        return alternative_prev if alternative_recall else prev

    map_orig = PIL.Image.open(MAP_PATH)
    drawer = PIL.ImageDraw.Draw(map_orig)
    coords = []
    colors = []

    for node1, node2 in zip(route, route[1:]):
        if node1 == alternative_edges_start:
            alternative_recall = True
        if node1 == alternative_edges_end:
            alternative_recall = False

        r1_vertex = all_node_locations[graph[node1]['giver'].lower()]
        r2_vertex = all_node_locations[graph[node2]['giver'].lower()]
        subroute = floyd_warshall_path(r1_vertex, r2_vertex, get_prev())

        c = get_closest_exterior_location(subroute[0]).coords
        t = to_map(c[0], c[1])
        drawer.ellipse((t[0] - 5,
                        t[1] - 5,
                        t[0] + 5,
                        t[1] + 5), fill=(0, 128, 0))

        if len(subroute) > 1:
            for i, r in enumerate(zip(subroute, subroute[1:])):

                r1, r2 = r
                try:
                    method, _ = get_edges()[r1][r2]
                except KeyError:
                    method = "Walk/Fly"

                c = get_closest_exterior_location(r1).coords
                t = to_map(c[0], c[1])
                coords.append((int(round(t[0])), int(round(t[1]))))
                colors.append(color_map[method])

    for c0, c1, col in zip(coords, coords[1:], colors):
        drawer.line([c0, c1], fill=col, width=2)
    map_orig.save(output_path)

draw_route(best, recall_edges, edges, 'mg_ajira_1', 'mg_edwinna_1', prev_recall, prev, "final_route_map.png")


# Test how the route improves if we have all Propylon indices
PROPYLONS = ['Valenvaryon', 'Rotheran', 'Indoranyon', 'Falensarano', 'Telasero', 'Marandus', 'Hlormaren', 'Andasreth', 'Berandas', 'Falasmaryon']
def add_propylon_edges(vertices, teleport_edges):
    new_edges = deepcopy(teleport_edges)
    propylon_vertices = [[v for v in vertices if '%s, Propylon Chamber' % p in v.cell_id][0] for p in PROPYLONS]
    for i in xrange(10):
        new_edges[propylon_vertices[i]][propylon_vertices[(i - 1) % 10]] = ('Propylon Teleport', 0)
        new_edges[propylon_vertices[i]][propylon_vertices[(i + 1) % 10]] = ('Propylon Teleport', 0)
    return new_edges

recall_propylon_edges = add_propylon_edges(vertices, recall_edges)
propylon_edges = add_propylon_edges(vertices, edges)
with open("dist_prop.txt", 'wb') as f:
    export_fw_data(propylon_edges, f)
with open("dist_recall_prop.txt", 'wb') as f:
    export_fw_data(recall_propylon_edges, f)

fw_dist = subprocess.Popen(["./floyd_warshall.exe", "dist_prop.txt", "dist_prop_res.txt", "prev_prop_res.txt"])
fw_dist_recall = subprocess.Popen(["./floyd_warshall.exe", "dist_prop_recall.txt", "dist_prop_recall_res.txt", "prev_prop_recall_res.txt"])
fw_dist.wait()
fw_dist_recall.wait()

dist_prop = np.loadtxt('dist_prop_res.txt')
prev_prop = np.loadtxt('prev_prop_res.txt', dtype=int)

dist_prop_recall = np.loadtxt('dist_recall_prop_res.txt')
prev_prop_recall = np.loadtxt('prev_recall_prop_res.txt', dtype=int)

node_distances_prop = make_node_distance_matrix(dist_prop)
node_distances_prop_recall = make_node_distance_matrix(dist_prop_recall)

with open("dist_graph_prop_norecall.txt", 'w') as f:
    export_optimiser_dists(node_distances_prop, graph, f)
with open("dist_graph_prop_recall.txt", 'w') as f:
    export_optimiser_dists(node_distances_prop_recall, graph, f)
args = [all_nodes.index('tt_set_sanctus_mark'), all_nodes.index('tt_endryn_1_end'), all_nodes.index('mg_ajira_1'), all_nodes.index('mg_edwinna_1')]
subprocess.check_call(["./ga_optimiser.exe", "dep_graph.txt", "dist_graph_prop_recall.txt", "dist_graph_prop_norecall.txt"] + [str(i) for i in ([
                       10000,  # pool size
                       1000,  # number of iterations
                       10,  # branching factor
                       5, ]  # number of random swaps to generate a new route
                      + args)])
with open("optimiser_result.txt.propylons", 'r') as f:
    pool = load_optimiser_pool(graph, f)
best = min(pool, key=evaluate_route)
with open("route_1mark_propylons.csv", 'wb') as f:
    export_route(best, recall_propylon_edges, propylon_edges, 'tt_set_sanctus_mark', 'tt_endryn_1_end', prev_prop_recall, prev_prop, f)
