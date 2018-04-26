import csv
from collections import defaultdict
from copy import copy

from yaml import load
from pprint import pprint

from dataloading import load_cells_npcs, Location
from travel import construct_graph, dijkstra, prune_graph, get_route, coalesce_cells, distance
import cPickle
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import random

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

inverted_graph = defaultdict(list)
for node, items in graph.items():
    for p in items.get('prerequisites', []):
        inverted_graph[p].append(node)

test = linearize(graph, start)
pprint(test)

# Load stuff from the Enchanted Editor dump
cells, npcs = load_cells_npcs("../Morrowind.esm.txt")

all_node_ids = set([i['giver'].lower() for i in graph.values()])
all_node_locations = {n.name.lower(): Location(n.position, cell=c) for c in cells for n in c.references if n.name.lower() in all_node_ids}

# Make sure we've located all POIs in the speedrun
for n in all_node_ids:
    if n not in all_node_locations:
        print("Failed to locate %s" % n)

vertices, edges = construct_graph(npcs, cells, extra_locations=all_node_locations.values(), use_realtime_metrics=True)
sorted_vertices = sorted(vertices)
vertex_index = {v: i for i, v in enumerate(sorted_vertices)}

def export_fw_data(edges, fd):
    INFINITY = 1e10

    for v1 in sorted_vertices:
        for v2 in sorted_vertices:
            d = distance(v1, v2, edges)
            fd.write('%f ' % (d if d is not None else INFINITY))
        fd.write('\n')


with open("dist.txt", 'wb') as f:
    export_fw_data(edges, f)

def floyd_warshall_path(v1, v2, prev):
    i1, i2 = vertex_index[v1], vertex_index[v2]

    if prev[i1, i2] is None:
        return []
    result = [sorted_vertices[i1]]
    while i1 != i2:
        i1 = prev[i1, i2]
        result.append(sorted_vertices[i1])
    return result

dist = np.loadtxt('dist_res.txt')
prev = np.loadtxt('prev_res.txt', dtype=int)

def floyd_warshall_distance(v1, v2, dist):
    return dist[vertex_index[v1], vertex_index[v2]]

floyd_warshall_distance(all_node_locations['vivec_god'], all_node_locations['trebonius artorius'], dist)
floyd_warshall_path(all_node_locations['vivec_god'], all_node_locations['trebonius artorius'], prev)

# create a matrix of node-to-node best distances
node_distances = defaultdict(dict)
for n1 in all_node_ids:
    for n2 in all_node_ids:
        node_distances[n1][n2] = floyd_warshall_distance(all_node_locations[n1], all_node_locations[n2], dist) if n1 != n2 else 0


node_matrix = []
for k, v in node_distances.iteritems():
    node_matrix.append([v[i] for i in node_distances])
import seaborn as sns
ax = sns.clustermap(node_matrix, yticklabels=node_distances.keys(), xticklabels=node_distances.keys(), figsize=(15, 15))
sns.set(font_scale=0.7)
plt.setp(ax.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(ax.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.savefig("travel_heatmap_cluster_new.png", dpi=200)


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

with open("dist_graph.txt", 'w') as f:
    export_optimiser_dists(node_distances, graph, f)
with open("dep_graph.txt", 'w') as f:
    export_optimiser_constraints(graph, f)

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
best = min(pool, key=evaluate_route)

def route_valid(route, start):
    if route[:len(start)] != start:
        return False
    completed = set()
    for r in route:
        for p in graph[r].get('prerequisites', []):
            if p not in completed:
                return False
        completed.add(r)
    return True


import random


def move(xs, i, j):
    if j < 0:
        return xs[:i + j] + [xs[i]] + xs[i + j:i] + xs[i + 1:]
    else:
        return xs[:i] + xs[i + 1:i + j + 1] + [xs[i]] + xs[i + j + 1:]


def mutate_route(route, start, mutations=1):
    successful_mutations = 0
    while True:
        i = random.randrange(0, len(route))
        j = 0
        while j == 0:
            j = random.randrange(-i, len(route) - i - 1)
        new_route = move(route, i, j)

        if route_valid(new_route, start):
            successful_mutations += 1
            if successful_mutations == mutations:
                return new_route


POOL_SIZE = 1000
ITERATIONS = 5
BRANCHING = 40
MUTATIONS = 30

pool = [mutate_route(test, start, MUTATIONS) for _ in xrange(POOL_SIZE)]

for i in range(ITERATIONS):
    pool = sorted(pool, key=evaluate_route, reverse=False)[:POOL_SIZE / BRANCHING]
    print ("iteration %d, best %.2f" % (i, evaluate_route(pool[0])))

    pool = [mutate_route(r, start, MUTATIONS) for _ in range(BRANCHING) for r in pool]
best = min(pool, key=evaluate_route)
# compare the two heatmaps (dijkstra/outsourced FW)


# Sanctus Mark and recall too far apart, blocking other usages of Mark
# High-leveled killings early before Blunt training/getting the Hammer
# need to calculate the amount of potions required (Health and Speed too, so that we have enough money and Alch skill. Fortify Strength too when we're about to fight?
# also consider the main damage dealer -- Blunt?
# finally, levelling issues -- might accidentally level things that we are in the middle of levelling. Perhaps worth pulling all training back?
# mg looting weirdness: get quest from aengoth and then do stuff around MG (including teleportation)



with open("speedrun_route_pool.bin", 'rb') as f:
    pool = cPickle.load(f)
# with open("speedrun_route_pool.bin", 'wb') as f:
#     cPickle.dump(pool, f)





# edges = [(p, n) for n in graph for p in graph[n].get('prerequisites', [])]
# import pydot
#
# G = pydot.graph_from_edges(edge_list=edges, directed=True)
#
# with open("quest_graph.png", 'wb') as f:
#     f.write(G.create_png(prog=['sfdp', '-Goverlap=false', '-Gsplines=true']))


def get_node_distance(n1, n2):
    return node_distances[graph[n1]['giver'].lower()][graph[n2]['giver'].lower()]




def blow_up_route(route):
    result = []

    for node1, node2 in zip(route, route[1:]):
        subroute = floyd_warshall_path(all_node_locations[graph[node1]['giver'].lower()], all_node_locations[graph[node2]['giver'].lower()], prev)
        result.append(subroute[:-1])

    return result

def get_best_mark_position(route):
    # cell_node_map = {v.cell: v for v in pruned_vertices}
    return min(
        (sum(get_node_distance(r1, r2) for r1, r2 in zip(route[:i], route[1:i]))
                           + sum(
            min(get_node_distance(r, r2), get_node_distance(r1, r2)) for r1, r2 in zip(route[i:], route[i + 1:])),
         i, r) for i, r in enumerate(route)
    )


route = best
big_route = blow_up_route(route)

total_route = [b for br in big_route for b in br]

#
# can't have marks between tt_set_sanctus_mark and tt_endryn_1_end
big_route_c = 0
for br, node in zip(big_route, route):
    if node == 'tt_set_sanctus_mark':
        forbidden_start = big_route_c
    if node == 'tt_endryn_1_end':
        forbidden_end = big_route_c

    big_route_c += len(br)

U = set(range(len(total_route)))
for i in xrange(forbidden_start, forbidden_end):
    U.remove(i)


def mutate_mark_arrangement(marks):
    marks = copy(marks)
    # randomly add/remove a mark
    if len(marks) == len(U):
        marks.remove(random.sample(marks, 1)[0])
    elif len(marks) == 0:
        marks.add(random.sample(U.difference(marks), 1)[0])
    elif random.random() > 0.5:
        marks.add(random.sample(U.difference(marks), 1)[0])
    else:
        marks.remove(random.sample(marks, 1)[0])
    return marks


def score_mark_arrangement(marks, route, big_route, dist):
    total_cost = 0
    last_mark = None
    big_route_counter = 0
    took_recall = False  # if we recalled from a previous point, we can't place a mark on the intermediate points we would have visited otherwise

    # Resulting route


    for r in zip([[]] + big_route, route, route[1:]): # make sure big_route doesn't look ahead
        br, r1, r2 = r
        if not took_recall:
            for i, m in enumerate(br):
                if i + big_route_counter in marks:
                    last_mark = m

        big_route_counter += len(br)

        r1_vertex = all_node_locations[graph[r1]['giver'].lower()]
        r2_vertex = all_node_locations[graph[r2]['giver'].lower()]

        if last_mark is not None and floyd_warshall_distance(r1_vertex, r2_vertex, dist) > floyd_warshall_distance(last_mark, r2_vertex, dist):
            total_cost += floyd_warshall_distance(last_mark, r2_vertex, dist)
            took_recall = True
        else:
            total_cost += floyd_warshall_distance(r1_vertex, r2_vertex, dist)
            took_recall = False

    return total_cost

#M = get_best_mark_position(best, big_route)
score_mark_arrangement([0], route, big_route, dist)

POOL_SIZE = 1000
ITERATIONS = 100
BRANCHING = 50
MUTATIONS = 30


def multiple_mark_mutation(marks, number=1):
    result = marks
    for _ in xrange(number):
        result = mutate_mark_arrangement(result)
    return result


pool = [multiple_mark_mutation(U, MUTATIONS) for _ in xrange(POOL_SIZE)]
for i in range(ITERATIONS):
    pool = sorted(pool, key=lambda m: score_mark_arrangement(m, route, big_route, dist), reverse=False)[:POOL_SIZE / BRANCHING]
    print ("iteration %d, best %.2f" % (i, score_mark_arrangement(pool[0], route, big_route, dist)))

    pool = [multiple_mark_mutation(r, MUTATIONS) for _ in range(BRANCHING) for r in pool]

best = min(pool, key=lambda m: score_mark_arrangement(m, route, big_route, dist))



# reconstruct route (have mark/recall + remove marks that are unused)

def export_route_with_marks(marks, route, dist, prev, fd):
    last_mark = None
    big_route_counter = 0
    new_big_route_counter = 0
    took_recall = False  # if we recalled from a previous point, we can't place a mark on the intermediate points we would have visited otherwise

    # Resulting route
    writer = csv.writer(fd)
    writer.writerow(['Route', 'Location', 'Description', 'Graph Node'])

    output = []

    for node1, node2 in zip(route, route[1:]):
        # import pdb; pdb.set_trace()
        r1_vertex = all_node_locations[graph[node1]['giver'].lower()]
        r2_vertex = all_node_locations[graph[node2]['giver'].lower()]
        graph_node = graph[node1]
        subroute = floyd_warshall_path(r1_vertex, r2_vertex, prev)


        output.append([str(subroute[0].cell) if subroute else '', graph_node.get('giver'), graph_node.get('description', node1), node1])

        new_big_route_counter += len(subroute)

        if last_mark is not None and floyd_warshall_distance(r1_vertex, r2_vertex, dist) > floyd_warshall_distance(last_mark, r2_vertex, dist):
            took_recall = True
            output.append(['Recall -> %s' % last_mark, '', '', ''])
            subroute = floyd_warshall_path(last_mark, r2_vertex, prev)
        else:
            took_recall = False

        if len(subroute) > 1:
            for i, r in enumerate(zip(subroute, subroute[1:])):
                r1, r2 = r
                try:
                    method, _ = edges[r1][r2]
                except KeyError:
                    method = "Walk/Fly"

                if (i + big_route_counter) in marks and not took_recall:
                    last_mark = r1
                    output.append(['Mark; %s -> %s' % (method, r2.cell), '', ''])
                else:
                    output.append(['%s -> %s' % (method, r2.cell), '', ''])
        big_route_counter = new_big_route_counter

    graph_node = graph[route[-1]]
    output.append([str(subroute[-1].cell) if subroute else '', graph_node.get('giver'), graph_node.get('description', node2), node2])

    # need to do a backwards pass through the output to eliminate spurious marks
    seen_recall = False
    for row in output[::-1]:
        if "Recall ->" in row[0]: # strictly speaking we should have an extra cell here or something
            seen_recall = True
        if "Mark; " in row[0]:
            if seen_recall:
                seen_recall = False
            else:
                row[0] = row[0][6:]

    for row in output:
        writer.writerow(row)

with open("route_marks.csv", 'wb') as f:
    export_route_with_marks(best, route, dist, prev, f)