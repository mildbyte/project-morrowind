import csv
from collections import defaultdict

from yaml import load
from pprint import pprint

from dataloading import load_cells_npcs, Location
from travel import construct_graph, dijkstra, prune_graph, coalesce_interiors, get_route


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

pruned_vertices, pruned_edges = prune_graph(vertices, edges, all_node_locations.values())
pruned_vertices, pruned_edges = coalesce_interiors(pruned_vertices, pruned_edges)

distprev = {}
for i, n in enumerate(all_node_ids):
    distprev[n] = dijkstra(pruned_vertices, pruned_edges, all_node_locations[n])
    print("%f%%" % (i / float(len(all_node_ids)) * 100))

import cPickle
with open("speedrun_dijkstra_dump.bin", 'rb') as f:
    distprev = cPickle.load(f)
with open("speedrun_dijkstra_dump.bin", 'wb') as f:
    cPickle.dump(distprev, f)


# get the corresponding vertices in the graph for each node
node_to_vertex = {}
for n in all_node_ids:
    for v in pruned_vertices:
        # after coalescing we're assuming there's one vertex per game cell
        if all_node_locations[n].cell == v.cell:
            node_to_vertex[n] = v

# create a matrix of node-to-node best distances
node_distances = defaultdict(dict)
node_routes = defaultdict(dict)
for n1 in all_node_ids:
    for n2 in all_node_ids:
        node_distances[n1][n2] = distprev[n1][0][node_to_vertex[n2]] if n1 != n2 else 0
        node_routes[n1][n2] = get_route(distprev[n1][1], node_to_vertex[n2]) if n1 != n2 else []


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
        return xs[:i+j] + [xs[i]] + xs[i+j:i] + xs[i+1:]
    else:
        return xs[:i] + xs[i+1:i+j+1] + [xs[i]] + xs[i+j+1:]


def mutate_route(route, start, mutations=1):
    successful_mutations = 0
    while True:
        i = random.randrange(0, len(route))
        j = 0
        while j == 0:
            j = random.randrange(-i, len(route)-i-1)
        new_route = move(route, i, j)

        if route_valid(new_route, start):
            successful_mutations += 1
            if successful_mutations == mutations:
                return new_route

POOL_SIZE = 10000
ITERATIONS = 1000
BRANCHING = 40
MUTATIONS = 30

pool = [mutate_route(test, start, MUTATIONS) for _ in xrange(POOL_SIZE)]

for i in range(ITERATIONS):
    pool = sorted(pool, key=evaluate_route, reverse=False)[:POOL_SIZE / BRANCHING]
    print ("iteration %d, best %.2f" % (i, evaluate_route(pool[0])))

    pool = [mutate_route(r, start, MUTATIONS) for _ in range(BRANCHING) for r in pool]


best = min(pool, key=evaluate_route)


def print_route(route):
    for node1, node2 in zip(route, route[1:]):
        graph_node = graph[node1]
        print(graph_node.get('giver'))
        print(graph_node.get('description', node1))
        print(node_routes[graph[node1]['giver'].lower()][graph[node2]['giver'].lower()])


def export_route(route, fd):
    writer = csv.writer(fd)
    writer.writerow(['Route', 'Location', 'Description', 'Graph Node'])

    for node1, node2 in zip(route, route[1:]):
        # import pdb; pdb.set_trace()
        graph_node = graph[node1]
        subroute = node_routes[graph[node1]['giver'].lower()][graph[node2]['giver'].lower()]

        writer.writerow([subroute[0].cell if subroute else '', graph_node.get('giver'), graph_node.get('description', node1), node1])

        if len(subroute) > 1:
            for r1, r2 in zip(subroute[1:], subroute[2:]):
                try:
                    method, _ = pruned_edges[r1][r2]
                except:
                    method = "Walk/Fly"

                writer.writerow(['%s -> %s' % (method, r2.cell), '', ''])

    graph_node = graph[route[-1]]
    writer.writerow([subroute[-1].cell if subroute else '', graph_node.get('giver'), graph_node.get('description', node2), node2])

with open("route.csv", 'wb') as f:
    export_route(best, f)



with open("speedrun_route_pool.bin", 'rb') as f:
    pool = cPickle.load(f)
with open("speedrun_route_pool.bin", 'wb') as f:
    cPickle.dump(pool, f)


edges = [(p, n) for n in graph for p in graph[n].get('prerequisites', [])]
import pydot
G = pydot.graph_from_edges(edge_list=edges, directed=True)

with open("quest_graph.png", 'wb') as f:
    f.write(G.create_png(prog=['sfdp', '-Goverlap=false', '-Gsplines=true']))