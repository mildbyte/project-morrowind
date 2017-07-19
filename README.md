# project-morrowind
Code that was used to create plots and images in my [project Morrowind series of blog posts](http://mildbyte.xyz/tags/morrowind.html) in which I load a dump of all locations and NPCs in Morrowind and do stuff like:

* draw a graph of all interiors in the game and how they're connected
* write a travel planner that [uses almost all of the means](https://imgur.com/fZxID) available in-game to minimize travel time
* draw a bubble chart of how long it takes to travel from a given location to every point in the game
* plot some population heatmaps

## Requirements
* Python 2.7 with stuff like NumPy/SciPy/matplotlib/PIL (I'll write a real setup.cfg soon, promise)
* A C++ compiler to turn `get_travel_times.cpp` to something useful
* The original Morrowind.esm file together with [Morrowind Enchanted Editor](http://mw.modhistory.com/download--1662) -- load the Morrowind.esm in the editor and then do File -> Dump to Text (with the default options, that is "Dump as record types, subrecord types and interpreted subrecord data" and "Dump in tab-indented tabular format"). I don't think it would be a good idea to provide the dump here since it's basically all the non-audio/model/sound game data and would probably get me in trouble.

