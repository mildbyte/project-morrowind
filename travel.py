# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:57:05 2017

@author: mildbyte
"""

from collections import deque, defaultdict
import heapq
import numpy as np
#import pygraphviz

from dataloading import Location


def generate_graph(cell_names, cell_destinations, cell_colors=None):
    G = pygraphviz.AGraph('graph G {}')
    G.node_attr['shape'] = 'box'
    G.node_attr['style'] = 'filled'
    G.graph_attr['splines'] = 'spline'
    G.graph_attr['overlap'] = 'prism'
    
    for n, d in zip(cell_names, cell_destinations):
        for dest in d:
            if not G.has_edge(n, dest):
                G.add_edge(n, dest)
    
    if cell_colors:
        for n, c in zip(cell_names, cell_colors):
            try:
                G.get_node(n).attr['fillcolor'] = c
            except KeyError:
                pass

    return G


def get_distance(v1, v2):
    return np.linalg.norm([c1 - c2 for c1, c2 in zip(v1, v2)])


def get_exterior_location_distance(l1, l2):
    l1 = get_closest_exterior_location(l1).coords
    l2 = get_closest_exterior_location(l2).coords
    
    return get_distance(l1, l2)


def get_closest_exterior_location(location):
    
    visited = set()
    q = deque([location])
    
    while q:
        l = q.pop()
        if l.cell_id in visited:
            continue
        visited.add(l.cell_id)
        
        if not l.cell.is_interior:
            return l
        
        q.extend(l.cell.destinations)

    raise AssertionError("Exterior unreachable from cell %s" % l.cell_id)


def get_closest_from(locations, source):
    return min(locations, key=lambda l: get_exterior_location_distance(l, source))

WALKING_SPEED = 1. / (30. / 100. / 3600.)

def distance(v1, v2, teleport_edges, walking_speed=WALKING_SPEED):
    # Custom distance function between two vertices that takes care of
    # teleportation (or travel by ship/silt strider) as well as walking
    # between two points in the same cell.
    if v2 in teleport_edges[v1]:
        return teleport_edges[v1][v2][1]
    if (v1.cell == v2.cell) or (not v1.cell.is_interior and not v2.cell.is_interior):
        return get_distance(v1.coords, v2.coords) / walking_speed
    else:
        return None


def dijkstra(vertices, teleport_edges, source):
    infinity = 10e50
    
    dist = {v: infinity for v in vertices}
    dist[source] = 0
    prev = {v: None for v in vertices}
    
    Q = []
    
    for v in vertices:
        heapq.heappush(Q, (dist[v], v))
    if source not in vertices:
        heapq.heappush(Q, (dist[source], source))
    
    while Q:
        _, v = heapq.heappop(Q)
        for v2 in vertices:
            if v != v2:
                d = distance(v, v2, teleport_edges)
                if d is None:
                    continue
                
                if dist[v] + d < dist[v2]:
                    dist[v2] = dist[v] + d
                    prev[v2] = v
                    #decrease-key
                    for i, node in enumerate(Q):
                        if node[1] == v2:
                            Q[i] = (dist[v] + d, v2)
                            break
                    heapq.heapify(Q)
    return dist, prev


def get_route(prev, start):
    result = []
    while start is not None and start in prev:
        result.append(start)
        start = prev[start]
    return list(reversed(result))


def construct_graph(npcs, cells, extra_locations=[], use_realtime_metrics=False, instant_travel_time=0.):
    print("Constructing the travel graph...")

    npc_id_map = {n.name: n for n in npcs}
    
    
    vertices = set(extra_locations)
    teleport_edges = defaultdict(dict)

    for c in cells:
        for r in c.references:
            ref_pos = Location(r.position, cell=c)
            if hasattr(r, 'destination'):
                # Add an instant-travel (in game and real time) teleport
                # between doors in different cells
                teleport_edges[ref_pos][r.destination] = ('Door', 0.)
                vertices.add(ref_pos)    
                vertices.add(r.destination)
            elif r.name in npc_id_map:
                if npc_id_map[r.name].destinations:
                    vertices.add(ref_pos)
                    for dest in npc_id_map[r.name].destinations:
                        # NPCs providing travel services: guild guides do
                        # instantaneous teleportation
                        if npc_id_map[r.name].class_name == 'Guild Guide':
                            teleport_edges[ref_pos][dest] = ('Guild Guide', instant_travel_time)
                        else:
                            # Ships/silt striders have a 16000-per-game hour
                            # travel speed (this is the only difference
                            # between realtime and gametime travel).
                            if use_realtime_metrics:
                                teleport_edges[ref_pos][dest] = ('Travel', instant_travel_time)
                            else:
                                teleport_edges[ref_pos][dest] = ('Travel', get_distance(ref_pos.coords, dest.coords) / 16000.)
                        vertices.add(dest)
    
    
    # Add 0-length edges from every point in the world to the nearest Temple
    # or Imperial Cult shrine (using spells)
    divine_marker_locations = {Location(n.position, cell=c) 
        for c in cells for n in c.references if n.name == "DivineMarker"}
    temple_marker_locations = {Location(n.position, cell=c)
        for c in cells for n in c.references if n.name == "TempleMarker"}
    
    vertices = vertices.union(divine_marker_locations).union(temple_marker_locations)
    for v in vertices:
        try:
            teleport_edges[v][get_closest_from(divine_marker_locations, v)] = ('Divine Intervention', 0.)
        except AssertionError:
            pass
        
        try:
            teleport_edges[v][get_closest_from(temple_marker_locations, v)] = ('Almsivi Intervention', 0.)
        except AssertionError:
            pass
    
    print("Done, %d vertices, %d teleport edges" % (len(vertices), sum(len(v) for v in teleport_edges.values())))
    return vertices, teleport_edges


def prune_graph(vertices, edges, pois):
    # Prunes the graph by removing all interiors that aren't on the route
    # from any given POI to the exterior
    poi_cells = set()

    curr_vertices = set(pois)
    visited = set()
    while curr_vertices:
        next_vertices = set()
        for v in curr_vertices:
            if v in visited:
                continue
            visited.add(v)
            # Stop exploring if we've reached an exterior
            if v.cell.is_interior:
                poi_cells.add(v.cell)
                for v2 in edges[v]:
                    next_vertices.add(v2)
                for v2 in vertices:
                    if v.cell == v2.cell:
                        next_vertices.add(v2)
        curr_vertices = next_vertices

    vertices = [v for v in vertices if v.cell in poi_cells or not v.cell.is_interior]

    new_edges = defaultdict(dict)
    for v1 in edges:
        if v1 in vertices:
            for v2 in edges[v1]:
                if v2 in vertices:
                    new_edges[v1][v2] = edges[v1][v2]

    return vertices, new_edges
    # also prune all exterior locations that are led to from unused cells or lead to unused cells


def coalesce_cells(vertices, edges):
    # Replaces all vertices in the graph in the same cell with a single one (average location)
    vertices_map = defaultdict(list)

    for v in vertices:
        vertices_map[v.cell].append(v)

    average_vertices = {}
    for cell, vs in vertices_map.items():
        coords = tuple(sum(v.coords[i] for v in vs) / float(len(vs)) for i in range(3))
        average_vertices[cell] = Location(coords=coords, cell_id=vs[0].cell_id, cell=vs[0].cell)

    new_vertices = set([average_vertices[v.cell] for v in vertices])

    grouped_edges = defaultdict(lambda: defaultdict(list))
    for v1 in edges:
        av1 = average_vertices[v1.cell]
        for v2 in edges[v1]:
            av2 = average_vertices[v2.cell]
            grouped_edges[av1][av2].append((edges[v1][v2][0], get_distance(av1.coords, v1.coords) / WALKING_SPEED + edges[v1][v2][1] + get_distance(v2.coords, av2.coords) / WALKING_SPEED))

    new_edges = defaultdict(dict)
    for av1 in grouped_edges:
        for av2 in grouped_edges[av1]:
            new_edges[av1][av2] = min(grouped_edges[av1][av2], key=lambda md: md[1])

    return new_vertices, new_edges