# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:50:32 2017

@author: mildbyte

Functions to load NPC/cell data from the Morrowind Enchanted
Editor text dump.
"""

import collections
import itertools
import struct


class IteratorWithLookahead(collections.Iterator):
    # Like a normal iterator but with the ability to peek at the next item
    # without consuming it (for when we need to know when the next record
    # starts so we can stop parsing the current one).
    def __init__(self, it):
        self.it, self.nextit = itertools.tee(iter(it))
        self._advance()
    def _advance(self):
        self.lookahead = next(self.nextit, None)
    def next(self):
        self._advance()
        return next(self.it)


def parse_destination(it):
    coords = [it.next()[1] for _ in xrange(3)] 
    for _ in xrange(3):
        it.next() #Ignore angle
    if it.lookahead[0] == "DNAM":
        return(Location(coords, it.next()[1]))
    else:
        return(Location(coords))


class Location():
    # Encodes a unique position in game (cell and coordinates in that cell).
    # Is supposed to be linked to an actual cell object after everything
    # is loaded (see finalize_references)
    def __init__(self, coords, cell_id=None, cell=None):
        self.coords = (float(coords[0]), float(coords[1]), float(coords[2]))
        self.cell_id = cell_id
        
        if cell:
            self.cell = cell
            self.cell_id = cell.get_full_name()
            
    def __repr__(self):
        return "(%s, (%f, %f, %f))" % (repr(self.cell), self.coords[0], self.coords[1], self.coords[2])
    
    def finalize_references(self, exterior_coord_map, cell_id_map):
        if not self.cell_id:
            x, y = self.coords[0], self.coords[1]
            cell_loc = "[%d,%d]" % (int(x/8192), int(y/8192))
            self.cell_id = exterior_coord_map[cell_loc]
            

        self.cell = cell_id_map[self.cell_id]
    
    # Some deduplication to make sure we don't look at the same point several
    # times (e.g. for pathfinding)
    def __hash__(self):
        return hash(self.coords) ^ hash(self.cell_id)
    
    def __eq__(self, l):
        return self.coords == l.coords and self.cell_id == l.cell_id


class NPC():
    def __init__(self, it):
        # Consumes an NPC record from the iterator
        self.name = it.next()[1]
        self.inventory = []
        self.destinations = []
        self.is_female = True
    
        while it.lookahead != None and it.lookahead[0] != "NAME":
            if it.lookahead[0] == "NPCO":
                count = int(it.next()[1])
                item_id = it.next()[1]
                self.inventory.append((item_id, count))
            elif it.lookahead[0] == "FNAM":
                self.full_name = it.next()[1]
            elif it.lookahead[0] == "RNAM":
                self.race = it.next()[1]
            elif it.lookahead[0] == "CNAM":
                self.class_name = it.next()[1]
            elif it.lookahead[0] == "FLAG":
                flags = it.next()[1]
                if flags == "":
                    continue
                flags = struct.unpack('<b', flags[0])
                    
                self.is_female = bool(flags[0] & 0x01)
            elif it.lookahead[0] == "DODT":
                self.destinations.append(parse_destination(it))
            else:
                it.next() #Ignore
    
    def finalize_references(self, exterior_coord_map, cell_id_map):
        # Make sure all the locations the NPC can take us to point to 
        # actual cell records
        for d in self.destinations:
            d.finalize_references(exterior_coord_map, cell_id_map)
    
    def __repr__(self):
        return "<NPC (%s, %s, %s, %s)>" % (self.name, self.full_name, self.race, self.class_name)


class CellReference():
    # Represents an instance of something in the cell (e.g. an NPC/a container etc)
    def __init__(self, it):
        it.next() #FRMR
        it.next() #FRMR
        self.name = it.next()[1] #NAME
        
        self.data = []
        self.is_deleted = False
        
        # Continue parsing the cell item until we've reached either another one
        # or a different cell/record
        while it.lookahead != None and it.lookahead[0] != "NAME" and it.lookahead[0] != "FRMR":
            if it.lookahead[0] == "DELE":
                self.is_deleted = True
                it.next()
            elif it.lookahead[0] == "DODT":
                # If it's a door, find out where it takes us
                self.destination = parse_destination(it)
            elif it.lookahead[0] == "DATA":
                x, y, z = [float(it.next()[1]) for _ in xrange(3)]
                [it.next() for _ in xrange(3)]
                self.position = (x, y, z)
            else:
                # Other data that we don't know how to parse but will keep just in case
                self.data.append(it.next())
        
    def finalize_references(self, exterior_coord_map, cell_id_map):
        try:
            self.destination.finalize_references(exterior_coord_map, cell_id_map)
        except AttributeError:
            pass
        except KeyError:
            pass


class Cell():
    def get_full_name(self):
        if (self.is_interior):
            return self.name
        else:
            if self.name != "":
                return "%s, [%d,%d]" % (self.name, self.grid[0], self.grid[1])
            try:
                return "%s, [%d,%d]" % (self.region_name, self.grid[0], self.grid[1])
            except AttributeError:
                return "Wilderness, [%d,%d]" % (self.grid[0], self.grid[1])
                
    def __init__(self, it):
        self.name = it.next()[1]
        self.references = []
        
        self.is_interior = struct.unpack('<h', it.next()[1])[0] % 2 == 1
        x = int(it.next()[1])
        y = int(it.next()[1])
        self.grid = (x, y)        
        
        while it.lookahead != None and it.lookahead[0] != "NAME":
            # Consume all things in the cell until we reach another cell/record
            if it.lookahead[0] == "RGNN": # Region name
                self.region_name = it.next()[1]
            elif it.lookahead[0] == "FRMR": # A cell reference
                ref = CellReference(it)
                if not ref.is_deleted:
                    self.references.append(ref)
            else:
                it.next()
                
        # Record everywhere we can go from this cell to make our lives easier
        self.destinations = [r.destination for r in self.references if hasattr(r, 'destination')]
    
    def __repr__(self):
        return self.get_full_name()
        
    def finalize_references(self, exterior_coord_map, cell_id_map):
        to_delete = []
        for d in self.destinations:
            try:
                d.finalize_references(exterior_coord_map, cell_id_map)
            except KeyError:
                to_delete.append(d)
        for r in self.references:
            r.finalize_references(exterior_coord_map, cell_id_map)
        
        for td in to_delete:
            self.destinations.remove(td)


def tokenize(raw):
    tokens = []
    for l in raw:
        t = l[:-2].split('\t')
        rec = t[1]
        dat = t[2].replace('<ESCAPE>', '\x1b').replace('<LINEFEED>', '\x0a')
        tokens.append((rec, dat))
    return tokens

    
def parse_cells(tokens):
    stream = IteratorWithLookahead(iter(tokens))
    cells = []
    while stream.lookahead != None:
        cells.append(Cell(stream))
    
    return cells


def parse_npcs(tokens):
    stream = IteratorWithLookahead(iter(tokens))
    npcs = []
    while stream.lookahead != None:
        npcs.append(NPC(stream))
    
    return npcs


def read_cells_npcs(path):
    cells = []
    npcs = []
    
    f = open(path, 'rb')
    
    for l in f:
        if l.startswith("CELL"):
            cells.append(l)
        if l.startswith("NPC_"):
            npcs.append(l)
    
    f.close()
    
    return cells, npcs


def load_cells_npcs(path):
    print "Reading the file..."
    cells, npcs = read_cells_npcs(path)
    
    print "Parsing the cells..."
    cells = parse_cells(tokenize(cells))
    
    print "Parsing the NPCs..."
    npcs = parse_npcs(tokenize(npcs))
    
    # Map the text cell IDs to actual data structures and link cells together
    exterior_names = {c.get_full_name() for c in cells if not c.is_interior}
    exterior_coord_map = {n.split()[-1]: n for n in exterior_names}
    cell_id_map = {c.get_full_name(): c for c in cells}
    
    print "Finalizing the cell references..."
    for c in cells:
        c.finalize_references(exterior_coord_map, cell_id_map)
    
    print "Finalizing the NPC references..."
    for n in npcs:
        n.finalize_references(exterior_coord_map, cell_id_map)
        
    return cells, npcs
