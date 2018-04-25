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
#
# pruned_vertices, pruned_edges = prune_graph(vertices, edges, all_node_locations.values())
# pruned_vertices, pruned_edges = coalesce_cells(pruned_vertices, pruned_edges)

def export_fw_data(vertices, edges, fd):
    INFINITY = 1e50

    vertices_ordered = sorted(vertices)
    for v1 in vertices_ordered:
        for v2 in vertices_ordered:
            d = distance(v1, v2, edges)
            fd.write('%f ' % (d if d is not None else INFINITY))
        fd.write('\n')

# with open("dist.txt", 'wb') as f:
#     export_fw_data(vertices, edges, f)



def floyd_warshall(vertices, edges):
    INFINITY = 1e50
    dist = {v1: {v2: INFINITY for v2 in vertices} for v1 in vertices}
    prev = {v1: {v2: None for v2 in vertices} for v1 in vertices}

    for v1 in vertices:
        for v2 in vertices:
            d = distance(v1, v2, edges)
            dist[v1][v2] = d if d is not None else INFINITY
            prev[v1][v2] = v2

    for i, v1 in enumerate(vertices):
        for v2 in vertices:
            for v3 in vertices:
                if dist[v2][v3] > dist[v2][v1] + dist[v1][v3]:
                    dist[v2][v3] = dist[v2][v1] + dist[v1][v3]
                    prev[v2][v3] = prev[v2][v1]
        print("%d/%d" % (i, len(vertices)))

    return dist, prev

def floyd_warshall_path(v1, v2, prev):
    if prev[v1][v2] is None:
        return []
    result = [v1]
    while not v1 == v2:
        v1 = prev[v1][v2]
        result.append(v1)
    return result

def load_vertex_data(sorted_vertices, fd):
    data = {}
    for v1, line in zip(sorted_vertices, fd.readlines()):
        data[v1] = {v2: float(l) for v2, l in zip(sorted_vertices, line.split(' '))}
    return data


dist = np.loadtxt()

with open("dist_res.txt", 'rb') as f:
    dist = load_vertex_data(sorted(vertices), f)

with open("prev_res.txt", 'rb') as f:
    prev = load_vertex_data(sorted(vertices), f)


# get the corresponding vertices in the graph for each node
node_to_vertex = {}
for n in all_node_ids:
    for v in pruned_vertices:
        # after coalescing we're assuming there's one vertex per game cell
        if all_node_locations[n].cell == v.cell:
            node_to_vertex[n] = v

# create a matrix of node-to-node best distances
node_distances = defaultdict(dict)
for n1 in all_node_ids:
    for n2 in all_node_ids:
        node_distances[n1][n2] = dist[all_node_locations[n1]][all_node_locations[n2]] if n1 != n2 else 0


node_matrix = []
for k, v in node_distances.iteritems():
    node_matrix.append([v[i] for i in node_distances])
import seaborn as sns
ax = sns.clustermap(node_matrix, yticklabels=node_distances.keys(), xticklabels=node_distances.keys(), figsize=(15, 15))
sns.set(font_scale=0.7)
plt.setp(ax.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(ax.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.savefig("travel_heatmap_cluster_new.png", dpi=200)


def evaluate_route(route):
    return sum(node_distances[graph[r1]['giver'].lower()][graph[r2]['giver'].lower()] for r1, r2 in zip(route, route[1:]))


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
#
#
# POOL_SIZE = 10000
# ITERATIONS = 5
# BRANCHING = 40
# MUTATIONS = 200
#
# pool = [mutate_route(test, start, MUTATIONS) for _ in xrange(POOL_SIZE)]
#
# for i in range(ITERATIONS):
#     pool = sorted(pool, key=evaluate_route, reverse=False)[:POOL_SIZE / BRANCHING]
#     print ("iteration %d, best %.2f" % (i, evaluate_route(pool[0])))
#
#     pool = [mutate_route(r, start, MUTATIONS) for _ in range(BRANCHING) for r in pool]


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



best = min(pool, key=evaluate_route)

# edges = [(p, n) for n in graph for p in graph[n].get('prerequisites', [])]
# import pydot
#
# G = pydot.graph_from_edges(edge_list=edges, directed=True)
#
# with open("quest_graph.png", 'wb') as f:
#     f.write(G.create_png(prog=['sfdp', '-Goverlap=false', '-Gsplines=true']))


def get_node_distance(n1, n2):
    return node_distances[graph[n1]['giver'].lower()][graph[n2]['giver'].lower()]


floyd_warshall_path(node_to_vertex['vivec_god'], node_to_vertex['trebonius artorius'], prev)

def blow_up_route(route):
    result = []

    for node1, node2 in zip(route, route[1:]):
        subroute = floyd_warshall_path(node_to_vertex[graph[node1]['giver'].lower()], node_to_vertex[graph[node2]['giver'].lower()], prev)
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

        r1_vertex = node_to_vertex[graph[r1]['giver'].lower()]
        r2_vertex = node_to_vertex[graph[r2]['giver'].lower()]

        if last_mark is not None and dist[r1_vertex][r2_vertex] > dist[last_mark][r2_vertex]:
            total_cost += dist[last_mark][r2_vertex]
            took_recall = True
        else:
            total_cost += dist[r1_vertex][r2_vertex]
            took_recall = False

    return total_cost

#M = get_best_mark_position(best, big_route)
score_mark_arrangement([0], route, big_route, dist)

POOL_SIZE = 10000
ITERATIONS = 10
BRANCHING = 20
MUTATIONS = 500


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

    cell_node_map = {v.cell.get_full_name(): v for v in pruned_vertices}

    output = []

    for node1, node2 in zip(route, route[1:]):
        # import pdb; pdb.set_trace()
        r1_vertex = node_to_vertex[graph[node1]['giver'].lower()]
        r2_vertex = node_to_vertex[graph[node2]['giver'].lower()]
        graph_node = graph[node1]
        subroute = floyd_warshall_path(r1_vertex, r2_vertex, prev)


        output.append([str(subroute[0].cell) if subroute else '', graph_node.get('giver'), graph_node.get('description', node1), node1])

        new_big_route_counter += len(subroute)

        if last_mark is not None and dist[r1_vertex][r2_vertex] > dist[last_mark][r2_vertex]:
            took_recall = True
            output.append(['Recall -> %s' % last_mark, '', '', ''])
            subroute = floyd_warshall_path(last_mark, r2_vertex, prev)
        else:
            took_recall = False

        if len(subroute) > 1:
            for i, r in enumerate(zip(subroute, subroute[1:])):
                r1, r2 = r
                try:
                    # import pdb; pdb.set_trace()
                    c1 = cell_node_map[r1.cell.get_full_name()]
                    c2 = cell_node_map[r2.cell.get_full_name()]
                    method, _ = pruned_edges[c1][c2]
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