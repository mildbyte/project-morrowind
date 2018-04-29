#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>


std::random_device rd;
std::mt19937 eng(rd());

bool validate_deps(const std::vector<int> &route, const std::vector<std::vector<int> > &deps) {
	// Check that the dependency graph is satisfied: maintain a mask of nodes that we've visited
	std::vector<bool> done(route.size());
	
	for (int i = 0; i < route.size(); i++) {
		for (int j = 0; j < deps[route[i]].size(); j++) {
			if (!done[deps[route[i]][j]]) return false;
		}
		done[route[i]] = true;
	}
	
	return true;
}

std::vector<int> mutate_route(std::vector<int> base, const std::vector<std::vector<int> > &deps, int start_hold_out, int no_mutations) {
	int successful = 0;
	// Random generator for the old and new position of a random point in the route (we're leaving first start_hold_out items alone)
	std::uniform_int_distribution<> rnd_size(start_hold_out, base.size() - 1);
	while (successful < no_mutations) {
		std::vector<int> old_base = base;
		int from = rnd_size(eng);
	
		int to = 0;
		do to = rnd_size(eng); while (to == from);
		
		// Move (not swap!) the item at "from" up/down towards "to"
		if (from < to) {
			std::rotate(base.begin() + from, base.begin() + from + 1, base.begin() + to);
		}
		else {
			std::rotate(base.begin() + to, base.begin() + from, base.begin() + from + 1);
		}
		
		if (validate_deps(base, deps)) {
			successful++;
		} else {
			// If dependencies not satisfied, throw this attempt away
			base = old_base;
		}
	}
	return base;
}

double evaluate_route(const std::vector<int> &route, const std::vector<std::vector<double> > &dist, const std::vector<std::vector<double> > &dist_norecall, int norecall_start, int norecall_end) {
	double total = 0.0;
	bool recalls_forbidden = false; //we're in between placing the Mark at Sanctus Shrine and actually doing the Silent Pilgrimage
	for (int i = 0; i < route.size() - 1; i++) {
		int node_id = route[i];
		if (node_id == norecall_start) recalls_forbidden = true;
		if (node_id == norecall_end) recalls_forbidden = false;
		
		if (recalls_forbidden) total += dist_norecall[route[i]][route[i+1]];
		else total += dist[route[i]][route[i+1]]; 
	}
	return total;
}

void topological_sort_visit(int node, const std::vector<std::vector<int> > &deps, std::vector<bool> &visited, std::vector<int> &result) {
	if (visited[node]) return;
	visited[node] = true;
	for (int i = 0; i < deps[node].size(); i++) {
		topological_sort_visit(deps[node][i], deps, visited, result);
	}
	result.push_back(node);
}

std::vector<int> topological_sort(const std::vector<std::vector<int> > &deps, const std::vector<int> &start) {
	std::vector<bool> visited(deps.size());
	std::vector<int> result;

	for (int i = 0; i < start.size(); i++) {
		visited[start[i]] = true;
		result.push_back(start[i]);
	}
	
	for (int i = 0; i < visited.size(); i++) {
		if (!visited[i])
			topological_sort_visit(i, deps, visited, result);
	}
	
	return result;
}

template <class T> void read_matrix(std::vector<std::vector<T> > &matrix, std::ifstream &source) {
	std::string input;
	while (std::getline(source, input)) {
		std::stringstream stream(input);
		std::vector<T> newl;
		T x;
		while (stream >> x) newl.push_back(x);
		matrix.push_back(newl);
	}
}

int main(int argc, char* argv[]) {
		
	if (argc < 10) {
		std::cerr << "Usage: " << argv[0] << " dependency_graph_file distance_graph_file non_recall_distance_graph_file pool_size no_iterations branching no_mutations recall_embargo_start_node recall_embargo_end_node [forced start nodes]" << std::endl;
		return -1;
	}
	
	std::vector<std::vector<double> > dist;
	std::vector<std::vector<double> > dist_nonrecall;
	std::ifstream source(argv[2]);
	std::cout << "Loading the recall-allowed distance matrix..." << std::endl;
	read_matrix<double>(dist, source);
	source.close();
	
	source.open(argv[3]);
	std::cout << "Loading the recall-not-allowed distance matrix..." << std::endl;
	read_matrix<double>(dist_nonrecall, source);
	source.close();
	
	source.open(argv[1]);
	std::vector<std::vector<int> > deps;
	
	std::cout << "Loading the dependency graph..." << std::endl;
	// Dependency graph: N lines, up to N dependencies in each line
	read_matrix<int>(deps, source);
	source.close();
	
	int pool_size = std::stoi(argv[4]);
	int no_iterations = std::stoi(argv[5]);
	int branching = std::stoi(argv[6]);
	int no_mutations = std::stoi(argv[7]);
	
	int recall_embargo_start_node = std::stoi(argv[8]);
	int recall_embargo_end_node = std::stoi(argv[9]);

	std::vector<int> start;
	for (int i = 0; i < argc - 10; i++) {
		start.push_back(std::stoi(argv[i + 10]));
	}
	
	// Initialize the solution pool
	std::vector<std::vector<int> > solution_pool;
	std::vector<int> initial = topological_sort(deps, start);

	for (int i = 0; i < pool_size; i++) {
		solution_pool.push_back(initial);
	}
	
	std::cout << "Optimising..." << std::endl;
	
	for (int i = 0; i < no_iterations; i++) {
		std::sort(solution_pool.begin(), solution_pool.end(),
			[&, dist, dist_nonrecall, recall_embargo_start_node, recall_embargo_end_node](std::vector<int> l, std::vector<int> r) {
				return evaluate_route(l, dist, dist_nonrecall, recall_embargo_start_node, recall_embargo_end_node)
				< evaluate_route(r, dist, dist_nonrecall, recall_embargo_start_node, recall_embargo_end_node);});
		double best = evaluate_route(solution_pool[0], dist, dist_nonrecall, recall_embargo_start_node, recall_embargo_end_node);
		std::cout << "Iteration " << i << "; best " << best << std::endl;
		
		std::vector<std::vector<int> > new_solution_pool;
		for (int i = 0; i < pool_size / branching; i++) {
			for (int j = 0; j < branching; j++) {
				new_solution_pool.push_back(mutate_route(solution_pool[i], deps, start.size(), no_mutations));
			}
		}
		solution_pool = new_solution_pool;
	}
	
	std::cout << "Optimisation done, exporting..." << std::endl;
	std::ofstream output("optimiser_result.txt");
	
	for (int i = 0; i < solution_pool.size(); i++) {
		for (int j = 0; j < solution_pool[i].size(); j++) {
			output << solution_pool[i][j] << " ";
		}
		output << std::endl;
	}
	output.close();

	return 0;
}